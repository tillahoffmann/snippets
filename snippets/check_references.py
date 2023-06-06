import argparse
from pathlib import Path
import re
from typing import List, Optional


BIB_PATTERN = re.compile(r"@\w+\{(.*?),")
BBL_PATTERN = re.compile(r"(?:\\bibitem(?:\[.*?\])?\{(.*?)\})|(?:\\entry\{(.*?)\})", re.S)


class Args:
    bib: Path
    bbl: Path


def __main__(arg_list: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="check for missing or unused references")
    parser.add_argument("bib", help="bibtex file containing references", type=Path)
    parser.add_argument("bbl", help="compiled bibliography of formatted references", type=Path)
    args: Args = parser.parse_args(arg_list)

    bib = args.bib.read_text()
    bib_refs = set(BIB_PATTERN.findall(bib))
    print(f"found {len(bib_refs)} bib references")

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


if __name__ == "__main__":  # pragma: no cover
    __main__()
