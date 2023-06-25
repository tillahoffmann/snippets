import argparse
from pathlib import Path
import re
from typing import List, Optional
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    import requests


BIB_PATTERN = re.compile(r"@\w+\{(.*?),")
BBL_PATTERN = re.compile(r"(?:\\bibitem(?:\[.*?\])?\{(.*?)\})|(?:\\entry\{(.*?)\})", re.S)
DOI_PATTERN = re.compile(r"doi\s*=\s*\{\s*(.*?)\s*\},")


class Args:
    bib: Path
    bbl: Path
    check_dois: bool


def __main__(arg_list: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="check for missing or unused references")
    parser.add_argument("bib", help="bibtex file containing references", type=Path)
    parser.add_argument("bbl", help="compiled bibliography of formatted references", nargs="?",
                        type=Path)
    parser.add_argument("--check-dois", action="store_true", help="check dois in the bib file")
    args: Args = parser.parse_args(arg_list)

    bib = args.bib.read_text()
    bib_refs = set(BIB_PATTERN.findall(bib))
    print(f"found {len(bib_refs)} bib references")

    # Check consistency of references if a bbl file is given.
    if args.bbl:
        bbl = args.bbl.read_text()
        bbl_refs = set(map(max, BBL_PATTERN.findall(bbl)))
        print(f"found {len(bbl_refs)} bbl references")

        extra = bib_refs - bbl_refs
        if extra:
            print(f"{len(extra)} extra references: {', '.join(sorted(extra))}")
        else:
            print("no extra references")

        missing = bbl_refs - bib_refs
        if missing:
            print(f"{len(missing)} missing references: {', '.join(sorted(missing))}")
        else:
            print("no missing references")

    # Check dois if desired.
    if args.check_dois:
        dois = re.findall(DOI_PATTERN, bib)
        for doi in dois:
            response = requests.get(f"https://dx.doi.org/{doi}", allow_redirects=False)
            if response.status_code != 302:
                print(f"doi {doi} could not be resolved")


if __name__ == "__main__":
    __main__()
