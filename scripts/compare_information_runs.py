"""Compare saved fresh information reports without contacting providers."""

import argparse
import json
from pathlib import Path

from open_video_summary.core.summarizers.information_reliability import compare_information_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    arguments = parser.parse_args()
    if len(arguments.reports) < 2:
        parser.error("Supply at least two independently generated reports.")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in arguments.reports]
    print(json.dumps(compare_information_reports(reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
