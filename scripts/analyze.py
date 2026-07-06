import json, glob, io, csv, os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import properscoring as ps
from matplotlib import path

REPO_DIR = Path(__file__).resolve().parent.parent   # solar-forecast/
PAYLOAD_DIR = REPO_DIR.parent / "Submitted_payloads"
CACHE_DIR = REPO_DIR / "data" / "eval_cache"
BERLIN = ZoneInfo("Europe/Berlin")

CHALLENGES = {
    4:  {"name": "Solar | point",    "kind": "point",
         "smard_module": 1004068, "smard_label": "Photovoltaics"},
    10: {"name": "Solar | quantile", "kind": "quantile",
         "quantiles": [0.025, 0.25, 0.5, 0.75, 0.975],
         "smard_module": 1004068, "smard_label": "Photovoltaics"},
    16: {"name": "Solar | ensemble", "kind": "ensemble", "n_members": 100,
         "smard_module": 1004068, "smard_label": "Photovoltaics"},

}
def submissions_target_date(path):
    """Parse the target date from the filename.

    Returns a naive `date` only. The filename encodes target_start as
    `2026-04-30T00_00_00_02_00` (colons -> underscores) and DROPS the offset
    sign, so the full tz-aware timestamp can't be reconstructed unambiguously
    from the name. For DE_LU the offset is always positive (+01:00/+02:00), so
    the calendar date is safe -- but if you ever need the exact instant or a
    non-positive offset, read `target_start` from the JSON instead.
    """
    stem = Path(path).name.split('__')[1][13:]   # 2026-04-30T00_00_00_02_00
    return datetime.strptime(stem[:10], "%Y-%m-%d").date()

def submission_dt(path):
    """Parse the leading submission timestamp from the filename (YYYYMMDD_HHMMSS_micro)."""
    stem = Path(path).name.split("__")[0]
    return datetime.strptime(stem[:22], "%Y%m%d_%H%M%S_%f")

def is_dry_run(path):
    return "dry_run" in str(path)
def index_challenge(ch_id: int) -> pd.DataFrame:
    """One row per payload file, parsed from the filename only (no JSON read).

    Note: `target_date` is a date, not a tz-aware timestamp -- see
    `submissions_target_date` for the offset-sign caveat.
    """
    rows = []
    for f in glob.glob(str(PAYLOAD_DIR / f"challenge_{ch_id}" / "*.json")):
        rows.append({
            "path": f,
            "submission_time": submission_dt(f),
            "target_date": submissions_target_date(f),
            "dry_run": is_dry_run(f)
        })
    return pd.DataFrame(
        rows,
        columns=["path", "submission_time", "target_date", "dry_run"],
    )


def forecast_median(path) -> pd.Series:
    """Median forecast as a UTC-indexed 96-step Series (opens the JSON).

    Handles all three payload shapes: point (96,), quantile (96,5) and
    ensemble (96,100). For 2-D arrays we take the median across axis 1, which
    is the q=0.5 column for quantiles and the member-median for ensembles.
    """
    d = json.load(open(path))
    vals = np.asarray(d["values"], dtype=float)
    med = vals if vals.ndim == 1 else np.median(vals, axis=1)
    idx = pd.date_range(d["target_start"], periods=96, freq="15min").tz_convert("UTC")
    return pd.Series(med, index=idx)


def fetch_smard(module_id, start_date, end_date) -> pd.Series:
    """UTC-indexed MW actuals for [start_date, end_date) (dates in Berlin local)."""
    cache = CACHE_DIR / f"smard_{module_id}_{start_date}_{end_date}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["value"]
    start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=BERLIN)
    end   = datetime(end_date.year,   end_date.month,   end_date.day,   tzinfo=BERLIN)
    payload = {"request_form": [{
        "format": "CSV", "moduleIds": [module_id], "region": "DE-LU",
        "timestamp_from": int(start.timestamp() * 1000),
        "timestamp_to":   int(end.timestamp()   * 1000),
        "type": "discrete", "language": "en", "resolution": "quarterhour"}]}
    r = requests.post("https://www.smard.de/nip-download-manager/nip/download/market-data",
                      json=payload, timeout=60)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig")), delimiter=";"))
    ts, vals = [], []
    for row in rows[1:]:
        raw = row[2].strip().replace(",", "") if len(row) >= 3 else ""
        if len(row) < 3 or not row[0].strip() or not raw or raw == "-":
            continue
        try:
            val = float(raw) * 4.0          # MWh/15min -> MW
        except ValueError:
            continue
        local = datetime.strptime(row[0].strip(), "%b %d, %Y %I:%M %p").replace(tzinfo=BERLIN)
        ts.append(local.astimezone(ZoneInfo("UTC")))
        vals.append(val)
    s = pd.Series(vals, index=pd.to_datetime(ts, utc=True)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s.to_frame("value").to_parquet(cache)
    return s


def collect_underforecasts(ch_id: int, start_date, end_date) -> pd.DataFrame:
    """Latest submission per target day in [start_date, end_date] whose daily
    mean bias (median forecast - actual) is negative (i.e. underforecasts).

    Filters on target_date, drops dry runs, and skips days whose actuals are
    not published yet.
    """
    spec = CHALLENGES[ch_id]
    idx = index_challenge(ch_id)
    idx = idx[~idx.dry_run]
    win = idx[(idx.target_date >= start_date) & (idx.target_date <= end_date)]
    win = win.sort_values("submission_time").groupby("target_date").tail(1)

    actual = fetch_smard(spec["smard_module"], start_date, end_date + timedelta(days=2))

    rows = []
    for r in win.itertuples():
        fc = forecast_median(r.path)
        a = actual.reindex(fc.index)
        if a.isna().any():
            continue                       # actuals not published yet -> skip day
        bias = float((fc - a).mean())
        if bias < 0:
            rows.append({"target_date": r.target_date,
                         "submission_time": r.submission_time,
                         "mean_bias": bias,
                         "path": r.path})
    return pd.DataFrame(
        rows,
        columns=["target_date", "submission_time", "mean_bias", "path"],
    ).sort_values("target_date")


def main():
    plt.rcParams["figure.figsize"] = (11, 4)
    plt.rcParams["axes.grid"] = True

    if not PAYLOAD_DIR.is_dir():
        raise FileNotFoundError(f"Payload directory not found: {PAYLOAD_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)



if __name__ == "__main__":
    main()