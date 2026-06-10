"""Unpack and check Spider 1.0 and the test-suite databases.

Google Drive often blocks scripted downloads of these files, so they are downloaded
by hand from the official links below and placed in data/raw/.
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from text2sql.manifest import sha256_file, utc_now, write_json
from text2sql.paths import DATA_DIR, RAW_DIR, SPIDER_DIR

SPIDER_URL = "https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view"
TESTSUITE_URL = "https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view"
SPIDER_FILES = ("train_spider.json", "train_others.json", "dev.json", "tables.json", "dev_gold.sql")
TESTSUITE_DIR = DATA_DIR / "testsuite" / "database"


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject absolute paths and '..' parts, so nothing is written outside the target."""
    members = []
    for info in zf.infolist():
        parts = PurePosixPath(info.filename).parts
        if info.filename.startswith(("/", "\\")) or ".." in parts or ":" in info.filename:
            sys.exit(f"unsafe path in zip: {info.filename}")
        if "__MACOSX" not in parts:
            members.append(info)
    return members


def extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def unpack_spider(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        members = safe_members(zf)
        tables = next(m for m in members if PurePosixPath(m.filename).name == "tables.json"
                      and len(PurePosixPath(m.filename).parts) <= 2)
        prefix = str(PurePosixPath(tables.filename).parent)
        prefix = "" if prefix == "." else prefix + "/"
        if SPIDER_DIR.exists():
            shutil.rmtree(SPIDER_DIR)
        for info in members:
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            # The hidden test files are never unpacked.
            if rel in SPIDER_FILES or rel.startswith("database/"):
                extract(zf, info, SPIDER_DIR / rel)
    files = {name: sha256_file(SPIDER_DIR / name) for name in SPIDER_FILES
             if (SPIDER_DIR / name).exists()}
    counts = {name: len(json.loads((SPIDER_DIR / name).read_text(encoding="utf-8")))
              for name in files if name.endswith(".json")}
    db_count = sum(1 for p in (SPIDER_DIR / "database").iterdir() if p.is_dir())
    return {"files_sha256": files, "record_counts": counts, "database_count": db_count}


def unpack_testsuite(zip_path: Path, keep: set[str]) -> dict:
    """Keep only the folders of the Spider dev databases."""
    if TESTSUITE_DIR.exists():
        shutil.rmtree(TESTSUITE_DIR)
    kept = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in safe_members(zf):
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or "database" not in parts:
                continue
            rel = parts[parts.index("database") + 1:]
            if len(rel) == 2 and rel[0] in keep and rel[1].endswith(".sqlite"):
                extract(zf, info, TESTSUITE_DIR.joinpath(*rel))
                kept += 1
    folders = sorted(p.name for p in TESTSUITE_DIR.iterdir())
    return {"sqlite_files": kept, "database_folders": len(folders),
            "missing_dev_databases": sorted(keep - set(folders))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spider-zip", type=Path, default=RAW_DIR / "spider_data.zip")
    parser.add_argument("--testsuite-zip", type=Path, default=RAW_DIR / "testsuitedatabases.zip")
    args = parser.parse_args()

    if not args.spider_zip.exists():
        sys.exit(f"Download spider_data.zip from {SPIDER_URL}\nand save it as {args.spider_zip}")
    print("unpacking Spider ...")
    source = {
        "dataset": "Spider 1.0",
        "license": "CC BY-SA 4.0",
        "url": SPIDER_URL,
        "zip_name": args.spider_zip.name,
        "zip_sha256": sha256_file(args.spider_zip),
        "unpacked_at": utc_now(),
        **unpack_spider(args.spider_zip),
    }
    dev_dbs = {row["db_id"] for row in
               json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))}
    if args.testsuite_zip.exists():
        print("unpacking test-suite databases ...")
        source["testsuite"] = {
            "url": TESTSUITE_URL,
            "zip_sha256": sha256_file(args.testsuite_zip),
            **unpack_testsuite(args.testsuite_zip, dev_dbs),
        }
    else:
        print(f"no test-suite zip; TS accuracy needs {TESTSUITE_URL}")
    write_json(SPIDER_DIR / "SOURCE.json", source)
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
