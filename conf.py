from docutils import nodes
from docutils.parsers.rst.states import Inliner, Struct
import inspect
from sphinx.application import Sphinx
from typing import Any, Dict, List, Optional, Tuple


project = "snippets"
master_doc = "README"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.shtest",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


def setup(app: Sphinx):
    app.add_role("docline", docline)


def docline(name: str, rawtext: str, text: str, lineno: int, inliner: Inliner,
            options: Optional[Dict[str, Any]] = None, content: Optional[List[str]] = None) \
        -> Tuple[List[nodes.Node], List[str]]:
    """
    Extract the first line of the docstring with nested parsing based on
    https://stackoverflow.com/a/68865718/1150961. See
    https://docutils.sourceforge.io/docs/howto/rst-roles.html#define-the-role-function for a
    description of the arguments.
    """
    options = options or {}
    content = content or []

    # Get the first line of the docstring.
    *modules, obj = text.split(".")
    modules = ".".join(modules)
    module = __import__(modules, fromlist=modules)
    obj = getattr(module, obj)
    docstring: Optional[str] = getattr(obj, "__doc__", None)
    if docstring is None:
        raise ValueError(f"{obj} does not have a docstring")
    line, *_ = docstring.strip().split("\n\n")
    line = line.replace("\n", " ")

    # Prepend the reference to the underlying object.
    if inspect.isfunction(obj):
        refrole = "func"
    else:
        refrole = "class"
    line = f":{refrole}:`~{text}`: {line}"

    memo = Struct(
        document=inliner.document,
        reporter=inliner.reporter,
        language=inliner.language,
    )
    parent = nodes.inline(rawtext, '', **options)
    processed, messages = inliner.parse(line, lineno, memo, parent)
    parent += processed
    return [parent], messages
