from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


FILES = [
    (
        "General Safety Guidelines.pdf",
        "https://www.osha.gov/sites/default/files/publications/OSHA3886.pdf",
    ),
    (
        "General safety and Health.pdf",
        "https://www.osha.gov/sites/default/files/publications/SMALL-BUSINESS.pdf",
    ),
    (
        "PH25-I-2 Citation and Order Writing Handbook 11_18_2025.pdf",
        "https://www.msha.gov/sites/default/files/Directive%20%26%20Guidance/Handbooks/PH25-I-2%20Citation%20and%20Order%20Writing%20Handbook%2011_18_2025.pdf",
    ),
    (
        "PH26-V-1 Electrical Inspection Procedures Handbook 8_12_2026.pdf",
        "https://www.msha.gov/sites/default/files/Directive%20%26%20Guidance/Handbooks/PH26-V-1%20Electrical%20Inspection%20Procedures%20Handbook%208_12_2026.pdf",
    ),
]


def main() -> None:
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES:
        destination = target_dir / filename
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)

    print(f"Done. Files saved to {target_dir.resolve()}")


if __name__ == "__main__":
    main()
