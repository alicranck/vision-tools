import sys
import types
from pathlib import Path

from vision_tools.utils.locations import (
    patch_ultralytics_asset_downloads,
    resolve_ultralytics_asset_path,
)


def test_resolve_ultralytics_asset_path_maps_bare_filename():
    resolved = resolve_ultralytics_asset_path(
        "mobileclip_blt.ts",
        Path("/tmp/cache/models"),
    )
    assert resolved == "/tmp/cache/models/mobileclip_blt.ts"


def test_resolve_ultralytics_asset_path_keeps_urls_and_paths():
    models_dir = Path("/tmp/cache/models")
    assert (
        resolve_ultralytics_asset_path(
            "https://example.com/mobileclip_blt.ts",
            models_dir,
        )
        == "https://example.com/mobileclip_blt.ts"
    )
    assert (
        resolve_ultralytics_asset_path(
            "/opt/models/mobileclip_blt.ts",
            models_dir,
        )
        == "/opt/models/mobileclip_blt.ts"
    )
    assert (
        resolve_ultralytics_asset_path(
            "nested/mobileclip_blt.ts",
            models_dir,
        )
        == "nested/mobileclip_blt.ts"
    )


def test_patch_ultralytics_asset_downloads_redirects_bare_assets(monkeypatch):
    ultralytics = types.ModuleType("ultralytics")
    ultralytics_utils = types.ModuleType("ultralytics.utils")
    ultralytics_downloads = types.ModuleType("ultralytics.utils.downloads")

    def _fake_attempt_download_asset(file, *args, **kwargs):
        return str(file)

    ultralytics_downloads.attempt_download_asset = _fake_attempt_download_asset
    ultralytics_utils.downloads = ultralytics_downloads
    ultralytics.utils = ultralytics_utils

    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", ultralytics_utils)

    patch_ultralytics_asset_downloads(Path("/tmp/cache/models"))

    assert (
        ultralytics_downloads.attempt_download_asset("mobileclip_blt.ts")
        == "/tmp/cache/models/mobileclip_blt.ts"
    )
    assert (
        ultralytics_downloads.attempt_download_asset("/tmp/existing/mobileclip_blt.ts")
        == "/tmp/existing/mobileclip_blt.ts"
    )
