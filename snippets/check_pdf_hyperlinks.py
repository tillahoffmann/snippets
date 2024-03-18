import argparse
from pathlib import Path
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
            "User-Agent": "snippets/check_pdf_hyperlinks",
            "Accept": "*/*",
        }

        for filename in args.filenames:
            pagenumbers_and_urls = set(get_urls(filename))
            print(f"found {len(pagenumbers_and_urls)} urls in {filename}")
            invalid_urls = {}
            valid_urls = set()
            for page, url in tqdm(sorted(pagenumbers_and_urls), desc="checking hyperlinks"):
                if url in invalid_urls:
                    valid = False
                elif url in valid_urls:
                    valid = True
                else:
                    try:
                        response = requests.get(
                            url,
                            allow_redirects=True,
                            headers=headers,
                        )
                        response.raise_for_status()
                        valid = True
                        valid_urls.add(url)
                    except (requests.HTTPError, requests.ConnectionError) as ex:
                        invalid_urls[url] = str(ex)
                        valid = False
                if not valid:
                    print(f"found invalid url {url} on page {page + 1}: {invalid_urls[url]}")
            print(f"found {len(invalid_urls)} invalid urls in {filename}")


if __name__ == "__main__":
    CheckPdfHyperlinks.run()
