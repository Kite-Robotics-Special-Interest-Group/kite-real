"""Download Alicia-D v5.6 MJCF/URDF/meshes and clone the vendor SDK.

Assets are intentionally not committed (see ``docs/alicia-d-native.md``).
The default destinations match ``configs/robot/alicia_d_v5_6.yaml``.

Examples:
    python scripts/fetch_alicia_assets.py
    python scripts/fetch_alicia_assets.py --proxy http://127.0.0.1:7897 --force
    python scripts/fetch_alicia_assets.py --skip-sdk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    "https://raw.githubusercontent.com/Synria-Robotics/"
    "Synria-Robot-Descriptions/main/synriard"
)
SDK_URL = "https://github.com/Synria-Robotics/Alicia-D-SDK"

DESCRIPTION_FILES = {
    "mjcf/Alicia_D_v5_6/"
    "Alicia_D_v5_6_gripper_50mm.xml": "Alicia_D_v5_6_gripper_50mm.xml",
    "urdf/Alicia_D_v5_6/"
    "Alicia_D_v5_6_gripper_50mm.urdf": "Alicia_D_v5_6_gripper_50mm.urdf",
}
MESH_NAMES = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "left_gripper_50mm",
    "right_gripper_50mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="获取 Alicia-D v5.6 机器人/SDK 资产")
    parser.add_argument("--dest", default=str(REPO_ROOT / "assets" / "synriard"))
    parser.add_argument(
        "--sdk-dest", default=str(REPO_ROOT / "third_party" / "Alicia-D-SDK")
    )
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"))
    parser.add_argument("--force", action="store_true", help="覆盖已有文件")
    parser.add_argument("--skip-sdk", action="store_true")
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


def _request(
    url: str, proxy: str | None, offset: int = 0
) -> urllib.response.addinfourl:
    headers = {"User-Agent": "kite-fetch-alicia/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    return opener.open(urllib.request.Request(url, headers=headers), timeout=60)


def download(url: str, destination: Path, proxy: str | None, retries: int) -> None:
    """Resume-friendly download; keeps a ``.part`` file between attempts."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and not part.exists():
        return
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            response = _request(url, proxy, offset=offset)
            mode = "ab" if offset and response.status == 206 else "wb"
            if mode == "wb":
                offset = 0
            with response, part.open(mode) as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            part.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts: {url}"
                ) from exc
            print(f"  retry {attempt}/{retries}: {exc}")
            time.sleep(min(2.0 * attempt, 8.0))


def valid_stl(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 84:
        return False
    with path.open("rb") as handle:
        head = handle.read(5)
    size = path.stat().st_size
    return head == b"solid" or (size - 84) % 50 == 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_descriptions(args: argparse.Namespace, dest: Path) -> list[Path]:
    paths: list[Path] = []
    for remote, name in DESCRIPTION_FILES.items():
        destination = dest / remote
        if args.force and destination.exists():
            destination.unlink()
        if not destination.exists():
            print(f"download {name}")
            download(f"{RAW_ROOT}/{remote}", destination, args.proxy, args.retries)
        paths.append(destination)
    mesh_dir = dest / "meshes" / "Alicia_D_v5_6" / "follower_standard"
    for name in MESH_NAMES:
        destination = mesh_dir / f"{name}.STL"
        if args.force and destination.exists():
            destination.unlink()
        if not valid_stl(destination):
            print(f"download {name}.STL")
            download(
                f"{RAW_ROOT}/meshes/Alicia_D_v5_6/follower_standard/{name}.STL",
                destination,
                args.proxy,
                args.retries,
            )
        if not valid_stl(destination):
            raise RuntimeError(f"invalid STL after download: {destination}")
        paths.append(destination)
    return paths


def clone_sdk(destination: Path) -> str | None:
    if destination.exists():
        return (
            subprocess.run(
                ["git", "-C", str(destination), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"clone {SDK_URL}")
    subprocess.run(
        ["git", "clone", "--depth", "1", SDK_URL, str(destination)],
        check=True,
    )
    return (
        subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        or None
    )


def main() -> int:
    args = parse_args()
    dest = Path(args.dest)
    paths = fetch_descriptions(args, dest)
    sdk_commit = None
    if not args.skip_sdk:
        sdk_commit = clone_sdk(Path(args.sdk_dest))
    manifest = {
        "source": RAW_ROOT,
        "sdk": None if args.skip_sdk else {"url": SDK_URL, "commit": sdk_commit},
        "files": {
            str(path.relative_to(dest)): {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        },
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"assets: {dest}")
    print(f"manifest: {manifest_path}")
    if sdk_commit:
        print(f"sdk commit: {sdk_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
