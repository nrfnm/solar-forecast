"""
Fetch monthly installed solar capacity (DC nameplate) from Fraunhofer ISE
Energy-Charts API and save to data/capacity_timeseries.csv.

Run once to generate the CSV, then copy to the VM:
    python scripts/fetch_capacity_ise.py
    scp data/capacity_timeseries.csv ubuntu@<vm>:~/solar-forecast/data/

After updating the CSV, also set in config.py:
    CAPACITY_MW = <latest value printed by this script>
"""

import requests
import pandas as pd
from pathlib import Path

_OUT = Path(__file__).parent.parent / "data" / "capacity_timeseries.csv"
_URL = "https://api.energy-charts.info/installed_power"


def fetch() -> pd.Series:
    resp = requests.get(
        _URL,
        params={"country": "de", "time_step": "monthly", "installation_decommission": "false"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    solar_dc = next(p for p in data["production_types"] if p["name"] == "Solar DC")

    # times are "MM.YYYY" strings → convert to month-start UTC timestamps
    index = pd.to_datetime(
        [f"{t[3:]}-{t[:2]}-01" for t in data["time"]], utc=True
    )
    # values are in GW → convert to MW
    values = [v * 1000.0 if v is not None else None for v in solar_dc["data"]]

    s = pd.Series(values, index=index, name="capacity_mw", dtype=float)
    return s.dropna()


if __name__ == "__main__":
    s = fetch()
    print(f"Coverage: {s.index[0].date()} → {s.index[-1].date()}")
    print(f"Latest:   {s.iloc[-1]:,.0f} MW  ({s.index[-1].strftime('%b %Y')})")
    print(f"Max:      {s.max():,.0f} MW")
    print("\nSet in config.py:")
    print(f"    CAPACITY_MW = {round(s.iloc[-1], -3):,.0f}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(_OUT, header=True)
    print(f"\nSaved → {_OUT}")
