import pytest
import re
from snippets.check_references import __main__
from typing import List
from unittest import mock


@pytest.mark.parametrize("bib, bbl, patterns", [
    (
        r"@article{key, ...} @book{other, ...}",
        r"",
        [r"extra references: key, other", r"no missing references"],
    ),
    (
        r"", r"\bibitem[{Bishop(2006)}]{Bishop2006}",
        [r"missing references: Bishop2006", r"no extra references"],
    ),
    (
        r"@article{key, ...} @book{other, ...}",
        r"\bibitem[{Bishop(2006)}]{Bishop2006} \bibitem{key}",
        [r"missing references: Bishop2006", r"extra references: other"]),
])
def test_check_references_bbl(bib: str, bbl: str, patterns: List[str],
                              capsys: pytest.CaptureFixture) -> None:
    with mock.patch("pathlib.Path.read_text", side_effect=[bib, bbl]):
        __main__(["bib-file", "bbl-file"])

    outerr = capsys.readouterr()
    for pattern in patterns:
        assert re.search(pattern, outerr.out)


def test_check_references(capsys: pytest.CaptureFixture) -> None:
    bib = """
    doi = {10.48550/arXiv.1804.06788},
    ...
    doi = {bla-bla},
    """
    with mock.patch("pathlib.Path.read_text", return_value=bib):
        __main__(["bib-file", "--check-dois"])

    outerr = capsys.readouterr()
    assert "10.48550/arXiv.1804.06788" not in outerr.out
    assert "bla-bla" in outerr.out
