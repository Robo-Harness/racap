import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.check_release import APPROVED_IMAGES, inspect
from scripts.prepare_assets import prepare


def _manifest(root, entries):
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/external_assets.json").write_text(json.dumps({"assets": entries}))


def test_release_check_rejects_credentials_and_external_data(tmp_path):
    _manifest(tmp_path, [{"path": "assets/states.bin"}])
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/states.bin").write_bytes(b"\x00data")
    (tmp_path / "configs/local.env").write_text("TOKEN=example")
    findings = inspect(tmp_path)
    assert "external/generated asset: assets/states.bin" in findings
    assert "local credential file: configs/local.env" in findings
    assert all("TOKEN=example" not in finding for finding in findings)


def test_asset_validation_precedes_all_writes(tmp_path):
    root, source = tmp_path / "release", tmp_path / "source"
    source.mkdir()
    (source / "good.bin").write_bytes(b"good")
    (source / "bad.bin").write_bytes(b"bad")
    _manifest(root, [
        {"path": "good.bin", "sha256": hashlib.sha256(b"good").hexdigest()},
        {"path": "bad.bin", "sha256": hashlib.sha256(b"expected").hexdigest()},
    ])
    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare(source, root=root)
    assert not (root / "good.bin").exists()


def test_asset_provisioning_never_overwrites_different_content(tmp_path):
    root, source = tmp_path / "release", tmp_path / "source"
    source.mkdir()
    (source / "asset.bin").write_bytes(b"expected")
    _manifest(root, [{"path": "asset.bin", "sha256": hashlib.sha256(b"expected").hexdigest()}])
    assert prepare(source, root=root) == 1
    assert prepare(source, root=root) == 0
    (root / "asset.bin").write_bytes(b"local changes")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare(source, root=root)
    assert (root / "asset.bin").read_bytes() == b"local changes"


def _release_scaffold(root):
    _manifest(root, [])
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/rats/LICENSE"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic fixture license")


@pytest.mark.parametrize("name", sorted(APPROVED_IMAGES))
def test_reviewed_documentation_image_is_allowed(tmp_path, name):
    _release_scaffold(tmp_path)
    target = tmp_path / name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / name, target)
    assert inspect(tmp_path) == []
    target.write_bytes(b"replaced image")
    assert f"unreviewed image content: {name}" in inspect(tmp_path)


def test_arbitrary_image_and_embedded_auth_are_rejected(tmp_path):
    _release_scaffold(tmp_path)
    (tmp_path / "rollout.png").write_bytes(b"image")
    (tmp_path / "sample.md").write_text("https://" + "user:synthetic-password@example.org")
    findings = inspect(tmp_path)
    assert "unreviewed file type: rollout.png" in findings
    assert "embedded URL credentials: sample.md:1" in findings
    assert all("synthetic-password" not in finding for finding in findings)


def test_tracked_scan_ignores_local_files_but_rejects_tracked_generated_data(tmp_path):
    _release_scaffold(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert inspect(tmp_path, tracked_only=True) == [
        "tracked scan has no files; stage the intended release first"
    ]
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    (tmp_path / "configs/local.env").write_text("SYNTHETIC_ONLY=1")
    assert inspect(tmp_path, tracked_only=True) == []
    assert "local credential file: configs/local.env" in inspect(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/log.txt").write_text("synthetic generated output")
    subprocess.run(["git", "-C", str(tmp_path), "add", "outputs/log.txt"], check=True)
    assert "generated/private directory: outputs/log.txt" in inspect(tmp_path, tracked_only=True)
