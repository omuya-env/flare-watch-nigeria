"""Download every upstream source file into data/raw and record a manifest of sizes and checksums."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

import config

RAW = config.RAW


def build_sources() -> list[dict]:
    sources = [
        {"name": "worldbank_sites", "url": config.WB_SITES_URL, "path": RAW / "worldbank_flare_sites.xlsx", "min_bytes": 5_000_000},
        {"name": "worldbank_country", "url": config.WB_COUNTRY_URL, "path": RAW / "worldbank_flare_country.xlsx", "min_bytes": 5_000},
        {"name": "boundaries", "url": config.BOUNDARIES_URL, "path": RAW / "nga_adm1.geojson", "min_bytes": 100_000},
    ]
    for year, name in config.EOG_FILES.items():
        sources.append(
            {"name": f"eog_{year}", "url": config.EOG_BASE + name, "path": RAW / "eog" / f"{year}.xlsx", "min_bytes": 50_000}
        )
    return sources


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, attempts: int = 3) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": config.USER_AGENT}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()
                temp = dest.with_suffix(dest.suffix + ".part")
                with open(temp, "wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                temp.replace(dest)
            return "downloaded"
        except Exception as error:  # noqa: BLE001 - any transport failure is retried
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(2 * attempt)
    return f"FAILED {last_error}"


def fetch_overpass(place_filter: str) -> str:
    south, west, north, east = config.OVERPASS_BBOX
    query = (
        '[out:json][timeout:180];node["place"~"^(' + place_filter + ')$"]'
        f"({south},{west},{north},{east});out body;"
    )
    headers = {"User-Agent": config.USER_AGENT}
    try:
        response = requests.post(config.OVERPASS_URL, data={"data": query}, headers=headers, timeout=300)
        response.raise_for_status()
        payload = response.json()
    except Exception as error:  # noqa: BLE001
        return f"FAILED {type(error).__name__}: {error}"
    if not payload.get("elements"):
        return "no elements returned"
    dest = RAW / "osm_settlements.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return f"downloaded {len(payload['elements'])} places"


def already_present(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if a cached copy exists")
    parser.add_argument("--skip-overpass", action="store_true", help="skip the nearest-settlement lookup")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    rows = []

    for source in build_sources():
        status = "cached"
        if args.force or not already_present(source["path"], source["min_bytes"]):
            status = fetch(source["url"], source["path"])
        if status == "FAILED" or status.startswith("FAILED"):
            rows.append((source["name"], status))
            manifest[source["name"]] = {"url": source["url"], "path": str(source["path"].relative_to(config.ROOT)), "status": status}
            continue
        if not already_present(source["path"], source["min_bytes"]):
            rows.append((source["name"], f"TOO SMALL {source['path'].stat().st_size if source['path'].exists() else 0}"))
            continue
        manifest[source["name"]] = {
            "url": source["url"],
            "path": str(source["path"].relative_to(config.ROOT)),
            "bytes": source["path"].stat().st_size,
            "sha256": sha256_of(source["path"]),
            "status": status,
        }
        rows.append((source["name"], f"{status}, {source['path'].stat().st_size / 1e6:.1f} MB"))

    if not args.skip_overpass:
        if already_present(RAW / "osm_settlements.json", 10_000):
            rows.append(("osm_settlements", "cached"))
        else:
            status = fetch_overpass("city|town|village")
            if status.startswith("FAILED") or status == "no elements returned":
                status = fetch_overpass("city|town")
            rows.append(("osm_settlements", status))
        manifest["osm_settlements"] = {"status": rows[-1][1], "bbox": config.OVERPASS_BBOX}

    width = max(len(name) for name, _ in rows)
    for name, status in rows:
        print(f"{name.ljust(width)}  {status}")

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest written to {(RAW / 'manifest.json').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()