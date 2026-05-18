import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "DA": {
        "base_url": "https://mis.nyiso.com/public/csv/damlbmp",
        "suffix": "damlbmp_zone_csv",
        "output_dir": Path("zone_data/DA"),
    },
    "RT": {
        "base_url": "https://mis.nyiso.com/public/csv/realtime",
        "suffix": "realtime_zone_csv",
        "output_dir": Path("zone_data/RT"),
    },
}


def month_starts(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield f"{year}{month:02d}01"


def download_file(url, destination):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with open(destination, "wb") as output_file:
            shutil.copyfileobj(response, output_file)


def import_archive(dataset_name, date_prefix, zip_dir):
    dataset = DATASETS[dataset_name]
    archive_name = f"{date_prefix}{dataset['suffix']}.zip"
    extract_dir = dataset["output_dir"] / f"{date_prefix}{dataset['suffix']}"
    archive_path = zip_dir / archive_name
    url = f"{dataset['base_url']}/{archive_name}"

    if extract_dir.exists() and any(extract_dir.glob("*.csv")):
        return "skipped"

    dataset["output_dir"].mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        download_file(url, archive_path)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    return "imported"

## Change the data depending on which year's data needs to be imported
def main():
    parser = argparse.ArgumentParser(description="Import NYISO DA and RT zone data.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("zone_data/_zip_cache"),
        help="Where downloaded zip archives are cached.",
    )
    args = parser.parse_args()

    totals = {"imported": 0, "skipped": 0, "failed": 0}
    for date_prefix in month_starts(args.start_year, args.end_year):
        for dataset_name in ("DA", "RT"):
            label = f"{dataset_name} {date_prefix[:4]}-{date_prefix[4:6]}"
            try:
                status = import_archive(dataset_name, date_prefix, args.zip_dir)
                totals[status] += 1
                print(f"{label}: {status}")
            except (urllib.error.URLError, zipfile.BadZipFile, OSError) as error:
                totals["failed"] += 1
                print(f"{label}: failed ({error})")

    print(
        "Done. "
        f"Imported {totals['imported']}, "
        f"skipped {totals['skipped']}, "
        f"failed {totals['failed']}."
    )


if __name__ == "__main__":
    main()
