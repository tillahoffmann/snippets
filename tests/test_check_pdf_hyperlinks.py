import colorama
import pytest
from snippets import check_pdf_hyperlinks


def remove_colors(text: str) -> str:
    for ansi_codes in [colorama.Fore, colorama.Back]:
        for code in vars(ansi_codes).values():
            text = text.replace(code, "")
    return text


def test_check_pdf_hyperlinks_error(capsys: pytest.CaptureFixture) -> None:
    check_pdf_hyperlinks.CheckPdfHyperlinks.run(
        ["tests/check_pdf_hyperlinks_error.pdf"]
    )
    out, _ = capsys.readouterr()
    out = remove_colors(out)
    assert "found 4 urls" in out
    assert "invalid url `https://tillahoffmann.github.io/404` on page 1" in out


def test_check_pdf_hyperlinks_ok(capsys: pytest.CaptureFixture) -> None:
    check_pdf_hyperlinks.CheckPdfHyperlinks.run(["tests/check_pdf_hyperlinks_ok.pdf"])
    out, _ = capsys.readouterr()
    out = remove_colors(out)
    assert "all 1 urls in out"
