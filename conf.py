from docutils import nodes
from docutils.parsers.rst.states import Inliner, Struct
import inspect
from snippets.util import get_first_docstring_paragraph
from sphinx.application import Sphinx
from typing import Any, Dict, List, Optional, Tuple


project = "snippets"
html_theme = "furo"
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
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
exclude_patterns = [
    "README.rst",
    "venv",
]
plot_include_source = True


def setup(app: Sphinx):
    app.add_role("docitem", docitem)


def docitem(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Optional[Dict[str, Any]] = None,
    content: Optional[List[str]] = None,
) -> Tuple[List[nodes.Node], List[str]]:
    """
    Extract the first paragraph of the docstring with nested parsing based on
    https://stackoverflow.com/a/68865718/1150961. See
    https://docutils.sourceforge.io/docs/howto/rst-roles.html#define-the-role-function
    for a description of the arguments.
    """
    options = options or {}
    content = content or []

    # Get the first line of the docstring.
    *modules, obj = text.split(".")
    modules = ".".join(modules)
    module = __import__(modules, fromlist=modules)
    obj = getattr(module, obj)
    paragraph = get_first_docstring_paragraph(obj)

    # Prepend the reference to the underlying object.
    if inspect.isfunction(obj):
        refrole = "func"
    else:
        refrole = "class"
    paragraph = f":{refrole}:`~{text}`: {paragraph}"

    memo = Struct(
        document=inliner.document,
        reporter=inliner.reporter,
        language=inliner.language,
    )
    parent = nodes.inline(rawtext, "", **options)
    processed, messages = inliner.parse(paragraph, lineno, memo, parent)
    parent += processed
    return [parent], messages
