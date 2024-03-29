import pytest
from snippets import check_pdf_hyperlinks


def test_check_pdf_hyperlinks(capsys: pytest.CaptureFixture) -> None:
    check_pdf_hyperlinks.CheckPdfHyperlinks.run(["tests/example.pdf"])
    out, _ = capsys.readouterr()
    assert "found 4 urls" in out
    assert "invalid url https://tillahoffmann.github.io/404 on page 1" in out
