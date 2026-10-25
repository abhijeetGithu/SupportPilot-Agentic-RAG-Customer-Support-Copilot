"""Download the public SearchUnify docs (docs.searchunify.com) as raw HTML.

    python scripts/crawl_docs.py [--out data/raw/docs] [--delay 0.5] [--refresh]

Uses the sitemap, respects robots.txt, keeps only article pages under /Content/
(skips images and archived versions), and skips files already downloaded.
Files are saved by their URL path, e.g. data/raw/docs/Content-Sources/Add.htm
"""

import argparse
import re
import time
import urllib.request
import urllib.robotparser
from pathlib import Path

BASE = "https://docs.searchunify.com/"
SITEMAP = BASE + "Sitemap-v2.xml"
USER_AGENT = "SupportPilot-portfolio-crawler/0.1 (personal non-commercial project)"
SKIP = re.compile(r"/(Resources[^/]*|GeneratedImages)/", re.IGNORECASE)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def list_pages() -> list[str]:
    xml = fetch(SITEMAP).decode("utf-8", errors="replace")
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    return sorted(
        {u for u in urls if "/Content/" in u and u.lower().endswith(".htm") and not SKIP.search(u)}
    )


def url_to_path(url: str, out: Path) -> Path:
    return out / url.split("/Content/", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/docs"))
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    parser.add_argument("--refresh", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    robots = urllib.robotparser.RobotFileParser(BASE + "robots.txt")
    robots.read()

    pages = list_pages()
    print(f"{len(pages)} article pages in sitemap")
    fetched = skipped = failed = 0
    for url in pages:
        path = url_to_path(url, args.out)
        if path.exists() and not args.refresh:
            skipped += 1
            continue
        if not robots.can_fetch(USER_AGENT, url):
            print(f"robots.txt disallows {url}")
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch(url))
            fetched += 1
        except (OSError, ValueError) as e:  # network/HTTP errors: keep going, report at the end
            print(f"FAILED {url}: {e}")
            failed += 1
        time.sleep(args.delay)
    print(f"fetched {fetched}, already present {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
