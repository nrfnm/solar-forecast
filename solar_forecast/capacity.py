"""
Generate k-Centroids from Marktstammdatenregister (MaStR)
"""
from open_mastr import Mastr
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import pandas as pd

_DB_PATH = Path(__file__).parent.parent/"open-mastr" / "data" / "MaStR_DB.sqlite"
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
    print(centroids.to_string(index=False))





