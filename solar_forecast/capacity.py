"""
Generate k-Centroids from Marktstammdatenregister (MaStR)

Data source: Marktstammdatenregister (MaStR), Bundesnetzagentur.
Licensed under Datenlizenz Deutschland – Namensnennung 2.0 (dl-de/by-2-0).
https://www.marktstammdatenregister.de
"""
from open_mastr import Mastr
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import pandas as pd

_DB_PATH = Path(__file__).parent.parent/"open-mastr" / "data" / "MaStR_DB.sqlite"
_CSV_PATH = Path(__file__).parent.parent / "data" / "capacity_timeseries.csv"
_PLZ_CSV = Path(__file__).parent.parent / "data" / "plz_geocoord.csv"
_REQUIRED_COLS = [
        "EinheitMastrNummer",
        "Breitengrad",
        "Laengengrad",
        "Nettonennleistung",
        "Bundesland",
        "Land",
    ]
_LOAD_COLS = _REQUIRED_COLS + ["Postleitzahl"]

def download_db(force = False):
    db_path = _DB_PATH
    db_path.parent.mkdir(exist_ok=True, parents=True)

    if db_path.exists() and not force:
        print(f"Database already exists at {db_path}")
        return

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        db = Mastr(engine)
        print("Downloading MaStR database...")
        db.download(data="solar")
    finally:
        engine.dispose()

def clean_db() -> pd.DataFrame:
    """
    Load MaStR solar data and return only active units with key columns.
    Coordinates missing from the MaStR record are filled via PLZ lookup.

    Returns
    -------
    pd.DataFrame
        Columns: EinheitMastrNummer, Breitengrad, Laengengrad,
        Nettonennleistung, Bundesland, Land.
        Only rows where EinheitBetriebsstatus == 'In Betrieb'.
    """
    engine = create_engine(f"sqlite:///{_DB_PATH}")
    try:
        cols = ", ".join(_LOAD_COLS)
        df = pd.read_sql(
            f"SELECT {cols} FROM solar_extended WHERE EinheitBetriebsstatus = 'In Betrieb'",
            engine
        )
    finally:
        engine.dispose()

    missing = [c for c in _LOAD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from MaStR table: {missing}")

    plz = pd.read_csv(_PLZ_CSV, index_col=0, dtype={"plz": str})
    plz.index = plz.index.astype(str).str.zfill(5)

    needs_coords = (df["Breitengrad"].isna() | df["Laengengrad"].isna()) & df["Postleitzahl"].notna()
    lookup_plz = df.loc[needs_coords, "Postleitzahl"].astype(str).str.zfill(5)
    df.loc[needs_coords, "Breitengrad"] = lookup_plz.map(plz["lat"]).values
    df.loc[needs_coords, "Laengengrad"] = lookup_plz.map(plz["lng"]).values

    return df[_REQUIRED_COLS].reset_index(drop=True)

def capacity_timeseries(tz: str = "UTC") -> pd.Series:
    """
    Build cumulative installed solar capacity (MW) per month from MaStR commissioning dates.

    Nettonennleistung is stored in kW in MaStR → divided by 1000 to get MW.

    Returns
    -------
    pd.Series
        Monthly timestamps (month-start frequency), values in cumulative MW.
        Timezone-aware as specified by tz.

    Raises
    ------
    FileNotFoundError
        If the MaStR DB has not been downloaded yet.
    """
    if _CSV_PATH.exists():
        s = pd.read_csv(_CSV_PATH, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = s.index.tz_convert(tz) if s.index.tz is not None else s.index.tz_localize(tz)
        return s.rename("capacity_mw")

    if _DB_PATH.exists() and _DB_PATH.stat().st_size > 0:
        engine = create_engine(f"sqlite:///{_DB_PATH}")
        try:
            df = pd.read_sql(
                "SELECT Inbetriebnahmedatum, Nettonennleistung "
                "FROM solar_extended "
                "WHERE EinheitBetriebsstatus = 'In Betrieb'",
                engine,
            )
        finally:
            engine.dispose()

        df["date"] = pd.to_datetime(df["Inbetriebnahmedatum"], errors="coerce")
        df["mw"] = pd.to_numeric(df["Nettonennleistung"], errors="coerce") / 1000.0
        df = df.dropna(subset=["date", "mw"])

        monthly = (
            df.set_index("date")["mw"]
            .resample("MS")
            .sum()
            .cumsum()
        )
        monthly.index = monthly.index.tz_localize(tz)
        return monthly.rename("capacity_mw")



    raise FileNotFoundError(
        f"Neither ISE Capacity CSV ({_CSV_PATH}) nor MaStR DB ({_DB_PATH}) found."
        "Run scripts/fetch_capacity_ise.py or run download_db()"
    )


def get_k_centroids(df: pd.DataFrame, k: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Cluster solar units into k representative grid points weighted by capacity.

    Returns
    -------
    pd.DataFrame
        Columns: lat, lon, weight (capacity fraction, sums to 1.0).
    """
    coords = df[["Breitengrad", "Laengengrad", "Nettonennleistung"]].dropna()

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(
        coords[["Breitengrad", "Laengengrad"]],
        sample_weight=coords["Nettonennleistung"],
    )

    coords = coords.copy()
    coords["cluster"] = km.labels_
    capacity_per_cluster = coords.groupby("cluster")["Nettonennleistung"].sum()
    total = capacity_per_cluster.sum()

    centers = pd.DataFrame(km.cluster_centers_, columns=["lat", "lon"])
    centers["weight"] = (capacity_per_cluster / total).values

    #fail loudly if centroids are not evenly distributed
    if centers["weight"].sum() != 1.0:
        raise ValueError("Centroid weights don't sum to 1.0")

    return centers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build capacity-weighted centroids from MaStR.")
    parser.add_argument("--k", type=int, default=5, help="Number of centroids (default: 5)")
    parser.add_argument("--force-download", action="store_true", help="Re-download MaStR even if DB exists")
    args = parser.parse_args()

    download_db(force=args.force_download)
    df = clean_db()
    centroids = get_k_centroids(df, k=args.k)

    print("# Paste into config.py:")
    print("CENTROIDS = [")
    for _, row in centroids.iterrows():
        print(f'    {{"lat": {row["lat"]:.4f}, "lon": {row["lon"]:.4f}, "weight": {row["weight"]:.4f}}},')
    print("]")





