import argparse
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
from typing import Generator, List, Optional, Tuple
from .util import get_first_docstring_paragraph, raise_for_missing_modules

with raise_for_missing_modules():
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
        headers = {
            # Some websites strictly check the user agent to be a browser.
            "User-Agent": "Mozilla/5.0 (Macintosh)",
            "Accept": "*/*",
        }

        for filename in args.filenames:
            pagenumbers_and_urls = set(get_urls(filename))
            print(f"found {len(pagenumbers_and_urls)} urls in {filename}")
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
                        parsed = urlparse(url)
                        response = requests.get(
                            url,
                            allow_redirects=not parsed.hostname.endswith("doi.org"),
                            headers=headers,
                        )
                        response.raise_for_status()
                        valid = True
                        valid_urls.add(url)
                    except (requests.HTTPError, requests.ConnectionError) as ex:
                        errors[url] = f"{ex.__class__.__name__}: {ex}"
                        valid = False
                if not valid:
                    print(f"found invalid url {url} on page {page + 1}: {errors[url]}")
            print(f"found {len(errors)} invalid urls in {filename}")


if __name__ == "__main__":
    CheckPdfHyperlinks.run()
