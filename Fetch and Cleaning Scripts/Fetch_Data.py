from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


BACKEND_CSV_URL = "https://climasphere-vk5q.onrender.com/download/downloadCSV"
ROOT_DIR = Path(__file__).resolve().parents[1]

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_csv(url: str, output_path: Path, timeout: int = 100) -> bool:
    if not is_valid_url(url):
        print("Invalid URL. Please provide a full http:// or https:// URL.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        request = Request(url, headers={"User-Agent": "climasphere-csv-downloader/1.0"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()

        if not data:
            print("Download finished, but the server returned an empty file.")
            return False

        temp_path.write_bytes(data)
        temp_path.replace(output_path)
        print(f"CSV saved successfully: {output_path}")
        return True

    except HTTPError as exc:
        print(f"Server returned an error: HTTP {exc.code} {exc.reason}")
    except URLError as exc:
        print(f"Could not connect to the URL: {exc.reason}")
    except TimeoutError:
        print("The request timed out. Please try again later.")
    except OSError as exc:
        print(f"Could not save the CSV file: {exc}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    if output_path.exists():
        print(f"Keeping existing CSV file unchanged: {output_path}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a CSV file without crashing on URL errors.")
    parser.add_argument("--url", default=BACKEND_CSV_URL, help="CSV download URL")
    parser.add_argument("--out", default=ROOT_DIR / "Datasets" / "Weather_Pollution_Data.csv", help="Output CSV path")
    parser.add_argument("--timeout", type=int, default=100, help="Network timeout in seconds")
    parser.add_argument("--strict-exit-code", action="store_true", help="Return exit code 1 when the download fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    success = download_csv(args.url, Path(args.out), args.timeout)
    return 0 if success or not args.strict_exit_code else 1


if __name__ == "__main__":
    raise SystemExit(main())
