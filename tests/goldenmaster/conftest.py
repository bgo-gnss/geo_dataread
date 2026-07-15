"""Session environment for the golden-master suite.

Renders a hermetic gpsconfig dir from the fixtures and points
GPS_CONFIG_PATH at it for the whole session, so no test ever touches
~/.config/gpsconfig or the production data mounts.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import build_config_dir  # noqa: E402


@pytest.fixture(scope="session")
def gm_env(tmp_path_factory):
    config_dir = build_config_dir(tmp_path_factory.mktemp("gpsconfig"))
    # Config dir WITHOUT the detrend CSV — the 0.4.x config-dir-relative
    # equivalent of the old "no CSV on CWD" fallback branch.
    empty_dir = build_config_dir(
        tmp_path_factory.mktemp("no-detrend-csv"), include_detrend=False
    )
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    yield {"config_dir": config_dir, "empty_dir": empty_dir}
    if old is None:
        os.environ.pop("GPS_CONFIG_PATH", None)
    else:
        os.environ["GPS_CONFIG_PATH"] = old
