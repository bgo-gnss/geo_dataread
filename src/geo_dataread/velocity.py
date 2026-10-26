"""The velocity product — moved to :mod:`geo_dataread.products.velocity_core`
(estimator), :mod:`~geo_dataread.products.velocity_files` (files) and
:mod:`~geo_dataread.products.velocity_cli` (``gps-velocities``) in gpsprod
step 3 (PLAN-gpsprod.md). Kept so every ``geo_dataread.velocity`` import
and test monkeypatch works unchanged: at run time this name IS the core
module (``sys.modules`` alias), so ``monkeypatch.setattr(vel, "_read", …)``
reaches the estimator; the imports below are the static view (mypy, IDEs).
"""

import sys

from geo_dataread.products import velocity_core as _core
from geo_dataread.products.velocity_cli import (  # noqa: F401
    _box,
    _date,
    _k_table_main,
    _show,
    _window,
    build_parser,
    main,
    select_stations,
)
from geo_dataread.products.velocity_core import *  # noqa: F403
from geo_dataread.products.velocity_core import (  # noqa: F401
    _HALF_DAY,
    _TERMS,
    DEFAULT_FRAME,
    DEFAULT_UNCERT,
    DEFAULT_VELOCITY_DIR,
    DEFAULT_VIEW,
    K_MIN_DAYS,
    K_RESIDUAL_REGIONAL,
    K_TABLE_SCHEMA,
    NO_RECORD_POLICIES,
    PRE_ONSET_BASE,
    READ_SCHEMAS,
    TIERS,
    _coords,
    _data_end,
    _fingerprint,
    _params_doc,
    _pos_dir,
    _product,
    _read,
    _record,
    _software,
    last_days_start,
)
from geo_dataread.products.velocity_files import (  # noqa: F401
    CSV_COLUMNS,
    _fmt,
    read_velocity_records,
    set_name,
    write_velocity_set,
)

__all__ = [
    *_core.__all__,
    "_box",
    "_date",
    "_k_table_main",
    "_show",
    "_window",
    "build_parser",
    "main",
    "select_stations",
    "_HALF_DAY",
    "_TERMS",
    "DEFAULT_FRAME",
    "DEFAULT_UNCERT",
    "DEFAULT_VELOCITY_DIR",
    "DEFAULT_VIEW",
    "K_MIN_DAYS",
    "K_RESIDUAL_REGIONAL",
    "K_TABLE_SCHEMA",
    "NO_RECORD_POLICIES",
    "PRE_ONSET_BASE",
    "READ_SCHEMAS",
    "TIERS",
    "_coords",
    "_data_end",
    "_fingerprint",
    "_params_doc",
    "_pos_dir",
    "_product",
    "_read",
    "_record",
    "_software",
    "last_days_start",
    "CSV_COLUMNS",
    "_fmt",
    "read_velocity_records",
    "set_name",
    "write_velocity_set",
]  # noqa: PLE0604

sys.modules[__name__] = _core
