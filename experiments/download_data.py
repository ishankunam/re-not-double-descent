"""Fetch/verify official Fashion-MNIST IDX files, or unpack bundled originals."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "FashionMNIST" / "raw"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
EXPECTED = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in EXPECTED.items():
        path = RAW / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            print(f"Downloading official {name}", flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(path)
        payload = path.read_bytes()
        md5 = hashlib.md5(payload).hexdigest()
        if md5 != expected:
            raise RuntimeError(f"Official checksum mismatch for {path}: {md5} != {expected}")
        uncompressed = path.with_suffix("")
        if not uncompressed.exists():
            with gzip.open(path, "rb") as source, uncompressed.open("wb") as target:
                shutil.copyfileobj(source, target)
        manifest[name] = {"md5": md5, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        print(f"Verified {name}", flush=True)
    (RAW.parent / "download_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
