"""ETFdb client for building ETF universe metadata."""

import json
import urllib.request
import pandas as pd

_DEFAULT_EXCLUDE = {"watchlist", "overall_rating"}

ETFDB_API_URL = "https://etfdb.com/api/screener/"


def _etfdb_post(payload):
    """POST a screener payload to ETFdb and return parsed JSON."""
    req = urllib.request.Request(ETFDB_API_URL, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8"), timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def available_filters():
    """Return ETFdb screener filter counts and values."""
    data = _etfdb_post({})
    return data.get("count", {})


def _parse_money_mm(text):
    """Parse ETFdb $MM strings into float (millions)."""
    if text is None:
        return float("nan")
    s = str(text).replace("$", "").replace(",", "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_number(text):
    """Parse numeric strings with commas/$/% into float."""
    if text is None:
        return float("nan")
    s = str(text).replace(",", "").replace("$", "").replace("%", "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _flatten_value(val):
    if isinstance(val, dict):
        return val.get("text", val)
    return val


def _normalize_record(rec, include_fields=None):
    data = {}
    for key, val in rec.items():
        if key in _DEFAULT_EXCLUDE:
            continue
        if include_fields is not None and key not in include_fields:
            continue
        data[key] = _flatten_value(val)
    return data


def fetch_top_by_category(category_field, top_n=10, sort_by="assets", per_page=50, include_fields=None):
    """Fetch top-N ETFs per category value from ETFdb screener.

    include_fields: optional list of fields to keep from ETFdb records.
    """
    counts = available_filters()
    if category_field not in counts:
        raise ValueError(f"Unknown category_field: {category_field}")
    values = list(counts[category_field].keys())
    rows = []
    for val in values:
        payload = {
            "page": 1,
            "per_page": max(per_page, top_n),
            "sort_by": sort_by,
            "sort_direction": "desc",
            category_field: [val],
        }
        resp = _etfdb_post(payload)
        records = resp.get("data", [])
        for rec in records[:top_n]:
            rec_flat = _normalize_record(rec, include_fields=include_fields)
            symbol = rec.get("symbol", {})
            name = rec.get("name", {})
            row = {
                "TICKER": _flatten_value(symbol),
                "NAME": _flatten_value(name),
                "AUM": _parse_money_mm(rec.get("assets")) * 1_000_000,
                "ADV": _parse_number(rec.get("average_volume")),
                "ASSET_CLASS": rec.get("asset_class"),
                "CATEGORY_TYPE": category_field,
                "CATEGORY": val,
                "SOURCE": "ETFDB",
            }
            row.update(rec_flat)
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(subset=["TICKER", "CATEGORY_TYPE", "CATEGORY"])


def build_universe(category_fields=("asset_class", "sizes", "investment_styles"), top_n=10, include_fields=None):
    """Build a combined ETF universe across category fields."""
    frames = []
    for field in category_fields:
        frames.append(fetch_top_by_category(field, top_n=top_n, include_fields=include_fields))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("expense_ratio", "net_expense_ratio"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_number)
    return df
