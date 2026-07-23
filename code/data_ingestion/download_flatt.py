"""Download Brookfield shareholder letters listed in a local manifest."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright


BASE = "https://bn.brookfield.com"
INDEX = BASE + "/reports-filings/letters-to-shareholders?tab={year}"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MANIFEST = SCRIPT_DIR / "flatt_download_manifest.json"
ARCHIVE_BASE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brookfield_shareholder_letters_2012_2026"
)


def key_from_text(year: int, text: str) -> str | None:
    text = " ".join(text.lower().split())
    if "march 2025 update" in text:
        return "march_update"
    if "2023 summary and update" in text:
        return "summary_update"
    for quarter, key in [
        ("first quarter", "q1"),
        ("second quarter", "q2"),
        ("third quarter", "q3"),
        ("fourth quarter", "q4"),
    ]:
        if quarter in text:
            return key
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Download manifest-listed letters and update their source metadata."""

    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lookup = {}

    for record in records:
        match = re.search(
            r"shareholder_letter_(\d{4})_"
            r"(q[1-4]|summary_update|march_update)\.pdf$",
            record["local_path"],
        )
        if match:
            lookup[(int(match.group(1)), match.group(2))] = record

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()

        for year in range(2012, 2027):
            page = context.new_page()
            page.goto(INDEX.format(year=year), wait_until="networkidle")

            for link in page.locator('main a[href*=".pdf"]:visible').all():
                href = link.get_attribute("href")
                if not href:
                    continue

                row_text = link.evaluate(
                    (
                        "element => (element.closest('tr') || "
                        "element.closest('li') || "
                        "element.parentElement).innerText"
                    )
                )
                key = key_from_text(year, row_text or "")
                record = lookup.get((year, key))

                if not record:
                    continue

                pdf_url = urljoin(BASE, href)
                output = PROJECT_ROOT / record["local_path"]
                output.parent.mkdir(parents=True, exist_ok=True)

                response = context.request.get(pdf_url)
                if not response.ok:
                    print("FAILED", year, key, response.status, pdf_url)
                    continue

                output.write_bytes(response.body())
                record["source_url"] = pdf_url
                record["sha256"] = sha256(output)
                print("SAVED", output.relative_to(PROJECT_ROOT))

            page.close()

        browser.close()

    MANIFEST.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(ARCHIVE_BASE),
        "zip",
        PROJECT_ROOT / "data" / "raw" / "investors",
        "flatt",
    )

    print("Manifest:", MANIFEST)
    print("ZIP:", archive)


if __name__ == "__main__":
    main()
