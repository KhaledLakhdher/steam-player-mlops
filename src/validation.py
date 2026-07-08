"""Data validation with pandera.

The schema is your *contract* with the data source. It catches malformed panel data — bad
genres, negative counts, wrong dtypes, missing columns — BEFORE it reaches training. This
matters most when you swap the synthetic generator for a real Kaggle CSV whose columns and
quality you don't control: garbage in fails loudly here instead of silently corrupting a model.
"""
from __future__ import annotations

try:
    import pandera.pandas as pa            # pandera >= 0.20
except ImportError:                         # older pandera
    import pandera as pa

import config

panel_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "game": pa.Column(str),
        "genre": pa.Column(str, pa.Check.isin(config.GENRES)),
        "days_since_release": pa.Column(int, pa.Check.ge(0)),
        "players": pa.Column(int, pa.Check.gt(0)),
    },
    strict=False,   # extra columns are allowed
    coerce=True,    # coerce dtypes where safely possible
)


def validate_panel(df):
    """Validate a raw panel against the schema.

    Raises pandera SchemaError(s) (lazy=True collects all failures) on bad data.
    Returns the validated (dtype-coerced) DataFrame on success.
    """
    return panel_schema.validate(df, lazy=True)
