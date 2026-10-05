# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify and unpack the frozen benchmark bundle without overwriting local work."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as handle:
        hasher = hashlib.sha256()
        while block := handle.read(1024 * 1024):
            hasher.update(block)
        return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout", type=Path, help="Target TRT-LLM checkout, mounted in the server container"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the archive and detect conflicts without writing",
    )
    args = parser.parse_args()
    root = args.checkout.resolve()
    if not (root / ".git").exists():
        parser.error("The destination must be an existing Git checkout")
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "manifest.json").read_text())
    archive = package / manifest["archive"]
    if digest(archive) != manifest["archive_sha256"]:
        parser.error("Archive checksum mismatch; retrieve artifacts.tar.gz with git lfs pull")
    expected = {item["path"]: item for item in manifest["files"]}
    missing = []
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive file list differs from manifest")
        for member in members:
            destination = (root / member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.name}")
            with bundle.extractfile(member) as handle:
                actual = hashlib.sha256(handle.read()).hexdigest()
            if actual != expected[member.name]["published_sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {member.name}")
            if destination.exists():
                if not destination.is_file() or digest(destination) not in (
                    expected[member.name]["original_sha256"],
                    expected[member.name]["published_sha256"],
                ):
                    raise FileExistsError(f"Local file differs; nothing overwritten: {destination}")
            else:
                missing.append(member)
        if not args.check_only:
            for member in missing:
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, destination.open("xb") as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
    print(
        f"Verified {len(members)} archived files; {len(missing)} missing files "
        f"{'would be restored' if args.check_only else 'restored'}; existing files preserved."
    )
    print(
        "The unpacker does not change source revisions, native libraries, environments, or servers."
    )


if __name__ == "__main__":
    main()
