import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from tqdm import tqdm
from typing import Generator, List, Optional, Tuple
from .util import get_first_docstring_paragraph, raise_for_missing_modules

with raise_for_missing_modules():
    import colorama
    import mechanize
    import pypdf
    import requests


class Args:
    filenames: List[Path]


def get_urls(filename: Path) -> Generator[Tuple[int, str], None, None]:
    """
    Extract page numbers and urls from a pdf document.
    """
    reader = pypdf.PdfReader(filename)
    for page in reader.pages:
        annotations = page.get("/Annots")
        if annotations is None:
            continue  # pragma: no cover
        for annotation in annotations.get_object():
            annotation = annotation.get_object()
            anchor = annotation.get("/A")
            if not anchor:
                continue  # pragma: no cover
            uri = anchor.get_object().get("/URI")
            if uri and uri.startswith("http"):
                yield (page.page_number, uri)


def validate_url(url):
    headers = {
        # Some websites strictly check the user agent to be a browser.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
        "Accept": "text/html,*/*",
    }

    try:
        parsed = urlparse(url)
        response = requests.get(
            url,
            allow_redirects=not parsed.hostname.endswith("doi.org"),
            headers=headers,
        )
        response.raise_for_status()
        return  # We successfully validated the url using requests.
    except (requests.HTTPError, requests.ConnectionError):
        # Failed to validate url using `requests`; trying `mechanize` ...
        pass

    # Try to send the request using a browser.
    browser = mechanize.Browser()
    browser.addheaders = list(headers.items())
    browser.set_handle_robots(False)
    response = browser.open(url)
    assert not response.errno


class CheckPdfHyperlinks:
    """
    Check LaTeX documents for missing or unused references.

    .. sh:: python -m snippets.check_pdf_hyperlinks --help
    """

    @classmethod
    def run(cls, argv: Optional[List[str]] = None) -> None:
        parser = argparse.ArgumentParser(description=get_first_docstring_paragraph(cls))
        parser = argparse.ArgumentParser()
        parser.add_argument("filenames", nargs="+", type=Path)
        args: Args = parser.parse_args(argv)

        for filename in args.filenames:
            pagenumbers_and_urls = set(get_urls(filename))
            print(f"found {len(pagenumbers_and_urls)} urls in `{filename}`")
            errors = {}
            valid_urls = set()
            for page, url in tqdm(
                sorted(pagenumbers_and_urls), desc="checking hyperlinks"
            ):
                if url in errors:
                    valid = False
                elif url in valid_urls:
                    valid = True
                else:
                    try:
                        validate_url(url)
                        valid = True
                        valid_urls.add(url)
                    except (
                        requests.HTTPError,
                        requests.ConnectionError,
                        URLError,
                    ) as ex:
                        errors[url] = f"{ex.__class__.__name__}: {ex}"
                        valid = False
                if not valid:
                    print(
                        f"{colorama.Fore.RED}found invalid url{colorama.Fore.RESET} "
                        f"`{url}` on page {page + 1}: {errors[url]}"
                    )
            if not errors:
                print(
                    f"{colorama.Fore.GREEN}all {len(pagenumbers_and_urls)} urls "
                    f"ok{colorama.Fore.RESET} in `{filename}`"
                )
            else:
                print(
                    f"{colorama.Fore.RED}{len(errors)} of {len(pagenumbers_and_urls)} "
                    f"urls are invalid{colorama.Fore.RESET} in `{filename}`"
                )


if __name__ == "__main__":
    CheckPdfHyperlinks.run()
