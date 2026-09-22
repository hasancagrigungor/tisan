"""
Gerçek veri setlerini ortak havuz formatına çevirir.

    python gercek_veri_hazirla.py            # data/raw -> data/havuz/gercek
    python gercek_veri_hazirla.py nab smd    # sadece seçilen kaynaklar
    python gercek_veri_hazirla.py lotsa      # LOTSA'yı HF'den indirir ve ekler (~7 GB)

Havuz formatı (uzun, kaynak başına bir Parquet):
    series_id  str      "smd/machine-1-1/test"
    timestamp  float64  unix saniye (kaynakta zaman yoksa sentetik, bkz. katalog)
    channel    int16    0..k-1
    value      float32  ham değer, NaN olabilir
    label      int8     1 = anomali, 0 = normal, -1 = etiketsiz

Seri bilgileri data/havuz/katalog.csv içinde: sektör, adım, satır/sütun sayısı,
etiket düzeyi (row: tüm sütunlara yayılmış satır etiketi, cell: sütun bazında),
lisans ve benchmark çakışması.

Kullanım:
    from gercek_veri_hazirla import seri_yukle
    s = seri_yukle("smd/machine-1-1/test")
    s["raw"]      # (T, 1+k)  kullanıcı biçimi: timestamp + değerler
    s["labels"]   # (T, k)    hücre etiketleri
"""
import ast
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "data"
RAW = ROOT / "raw"
OUT = ROOT / "havuz" / "gercek"
KATALOG = ROOT / "havuz" / "katalog.csv"

# TSB-AD değerlendirme setinde de bulunan kaynaklar. README §9: benchmark verisi eğitime girmez.
# Eğitimde kullanılacaksa ilgili benchmark bölümü değerlendirmeden çıkarılmalı.
BENCHMARK = {"nab": "TSB-AD-U", "smap": "TSB-AD-M", "msl": "TSB-AD-M", "smd": "TSB-AD-M",
             "skab": "SKAB benchmark"}

LISANS = {"skab": "AGPL-3.0", "skab_teaser": "AGPL-3.0", "nab": "AGPL-3.0",
          "smap": "telemanom (Apache-2.0), veri NASA", "msl": "telemanom (Apache-2.0), veri NASA",
          "pump": "belirsiz (Kaggle: unknown)", "smd": "MIT", "cnc": "CC0-1.0",
          "wind_gearbox": "Apache-2.0", "hai": "CC-BY-SA-4.0"}


def _unix(s):
    return pd.to_datetime(s).astype("int64").to_numpy() / 1e9


def _synthetic_time(T, step, start=1_577_836_800.0):   # 2020-01-01
    return start + step * np.arange(T, dtype=np.float64)


def _seri(sid, source, sector, t, X, labels, names, label_level, synthetic_time, **meta):
    """Tek seriyi (uzun tablo, katalog satırı) olarak döndürür."""
    X = np.asarray(X, dtype=np.float32)
    T, k = X.shape
    labels = np.asarray(labels, dtype=np.int8)
    labels = np.broadcast_to(labels if labels.ndim != 1 else labels[:, None], (T, k))
    order = np.argsort(t, kind="stable")
    t, X, labels = np.asarray(t, dtype=np.float64)[order], X[order], labels[order]
    long = pd.DataFrame({
        "series_id": sid,
        "timestamp": np.repeat(t, k),
        "channel": np.tile(np.arange(k, dtype=np.int16), T),
        "value": X.ravel(),
        "label": labels.ravel(),
    })
    row = labels.max(axis=1)
    kat = dict(series_id=sid, source=source, sector=sector, n_rows=T, n_channels=k,
               step_s=float(np.median(np.diff(t))) if T > 1 else np.nan,
               label_level=label_level,
               anomaly_row_ratio=float((row == 1).mean()) if (row >= 0).any() else np.nan,
               unlabeled=bool((labels == -1).all()),
               synthetic_time=synthetic_time,
               benchmark=BENCHMARK.get(source, ""), license=LISANS.get(source, ""),
               channel_names="|".join(map(str, names)), **meta)
    return long, kat


# =============================================================================
# Kaynaklar
# =============================================================================
def load_skab():
    base = RAW / "skab" / "SKAB"
    for f in sorted(base.rglob("*.csv"), key=lambda p: (p.parent.name, int(p.stem) if p.stem.isdigit() else 0)):
        d = pd.read_csv(f, sep=";")
        names = [c for c in d.columns if c not in ("datetime", "anomaly", "changepoint")]
        lab = d["anomaly"].fillna(0).astype(int).to_numpy() if "anomaly" in d else np.zeros(len(d), int)
        yield _seri(f"skab/{f.parent.name}/{f.stem}", "skab", "manufacturing", _unix(d["datetime"]),
                    d[names].to_numpy(), lab, names, "row", False, note="su pompası test düzeneği")


def load_skab_teaser():
    d = pd.read_csv(RAW / "skab_teaser" / "SKAB teaser.csv", sep=";")
    w = d.pivot_table(index="datetime", columns="id", values="value", aggfunc="mean").sort_index()
    yield _seri("skab_teaser/0", "skab_teaser", "manufacturing", _unix(w.index), w.to_numpy(),
                -1, list(w.columns), "row", False, note="etiketsiz, SKAB düzeneği")


def load_nab():
    base = RAW / "nab"
    windows = json.load(open(base / "labels" / "combined_windows.json"))
    sector = {"realAWSCloudwatch": "it", "realAdExchange": "advertising", "realTraffic": "transport",
              "realTweets": "social_media", "artificialNoAnomaly": "abstract",
              "artificialWithAnomaly": "abstract"}
    known = {"ambient_temperature_system_failure": "environment", "nyc_taxi": "transport",
             "machine_temperature_system_failure": "manufacturing",
             "cpu_utilization_asg_misconfiguration": "it", "ec2_request_latency_system_failure": "it",
             "rogue_agent_key_hold": "it", "rogue_agent_key_updown": "it"}
    for key, wins in sorted(windows.items()):
        cat, name = key.split("/")
        d = pd.read_csv(base / cat / cat / name)
        t = _unix(d["timestamp"])
        lab = np.zeros(len(d), dtype=np.int8)
        for a, b in wins:
            lab[(t >= _unix(pd.Series([a]))[0]) & (t <= _unix(pd.Series([b]))[0])] = 1
        sec = known.get(name[:-4], sector.get(cat, "unknown"))
        yield _seri(f"nab/{cat}/{name[:-4]}", "nab", sec, t, d[["value"]].to_numpy(), lab, ["value"],
                    "row", False, note="etiket = NAB anomali pencereleri")


def load_smap_msl():
    base = RAW / "smap_msl"
    meta = pd.read_csv(base / "labeled_anomalies.csv")
    # P-2 kanalı dosyada iki satır: anomali aralıkları birleştirilir
    meta = (meta.assign(anomaly_sequences=meta.anomaly_sequences.map(ast.literal_eval))
                .groupby("chan_id", as_index=False).agg(spacecraft=("spacecraft", "first"),
                                                        anomaly_sequences=("anomaly_sequences", lambda x: sum(x, []))))
    for _, r in meta.sort_values(["spacecraft", "chan_id"]).iterrows():
        src = r.spacecraft.lower()
        for split in ("train", "test"):
            X = np.load(base / "data" / "data" / split / f"{r.chan_id}.npy")
            T, k = X.shape
            lab = np.full((T, k), -1 if split == "train" else 0, dtype=np.int8)
            if split == "test":
                lab[:, 1:] = 0
                for a, b in r.anomaly_sequences:
                    lab[a:b + 1, 0] = 1                      # anomali telemetri sütununda
            names = ["telemetry"] + [f"cmd_{i}" for i in range(1, k)]
            yield _seri(f"{src}/{r.chan_id}/{split}", src, "space", _synthetic_time(T, 60.0), X, lab,
                        names, "cell", True,
                        note="sütun 0 telemetri, diğerleri komut kodlaması; train etiketsiz (normal varsayılır)")


def load_pump():
    d = pd.read_csv(RAW / "pump" / "sensor.csv", index_col=0)
    names = [c for c in d.columns if c.startswith("sensor_") and d[c].notna().any()]   # sensor_15 tamamen boş
    lab = d["machine_status"].isin(["BROKEN", "RECOVERING"]).astype(np.int8).to_numpy()
    yield _seri("pump/0", "pump", "manufacturing", _unix(d["timestamp"]), d[names].to_numpy(), lab, names,
                "row", False, note="etiket = BROKEN veya RECOVERING (7 arıza)")


def load_smd():
    base = RAW / "smd" / "ServerMachineDataset"
    for f in sorted((base / "test").glob("*.txt")):
        m = f.stem
        for split in ("train", "test"):
            X = np.loadtxt(base / split / f"{m}.txt", delimiter=",")
            T, k = X.shape
            if split == "train":
                lab = np.full((T, k), -1, dtype=np.int8)
            else:
                row = np.loadtxt(base / "test_label" / f"{m}.txt").astype(np.int8)
                lab = np.zeros((T, k), dtype=np.int8)
                covered = np.zeros(T, dtype=bool)
                for line in (base / "interpretation_label" / f"{m}.txt").read_text().split():
                    rng_, chans = line.split(":")
                    a, b = map(int, rng_.split("-"))
                    cols = [int(c) - 1 for c in chans.split(",")]   # dosyada 1'den başlıyor
                    lab[a:b + 1, cols] = 1
                    covered[a:b + 1] = True
                lab[(row == 1) & ~covered] = 1                  # açıklaması olmayan anomali: tüm sütunlar
                lab[row == 0] = 0
            if split == "train":
                t0 = T                                          # test, train'in devamı: zaman kesintisiz akar
            yield _seri(f"smd/{m}/{split}", "smd", "it", _synthetic_time(T, 60.0, 1_577_836_800.0 + 60 * (0 if split == "train" else t0)),
                        X, lab, [f"metric_{i}" for i in range(k)], "cell", True,
                        note="min-max normalize edilmiş sunucu metrikleri; train etiketsiz")


def load_cnc():
    base = RAW / "cnc"
    info = pd.read_csv(base / "train.csv").set_index("No")
    for f in sorted(base.glob("experiment_*.csv")):
        d = pd.read_csv(f)
        no = int(f.stem.split("_")[1])
        names = [c for c in d.columns if c != "Machining_Process" and pd.api.types.is_numeric_dtype(d[c])]
        r = info.loc[no]
        yield _seri(f"cnc/experiment_{no:02d}", "cnc", "manufacturing", _synthetic_time(len(d), 0.1),
                    d[names].to_numpy(), -1, names, "row", True,
                    note=f"etiketsiz; takım={r.tool_condition}, tamamlandı={r.machining_finalized}, "
                         f"görsel={r.passed_visual_inspection}")


def load_wind_gearbox():
    """Rüzgar türbini dişli kutusu SCADA (aiwithcagri, Kaggle): 10 dk, 5 yıl, 7 sensör."""
    base = RAW / "wind_gearbox"
    for f in ("labeled", "complex"):
        d = pd.read_csv(base / f"turbine_5yr_{f}_data.csv")
        names = [c for c in d.columns if c not in ("timestamp", "is_anomaly")]
        lab = d["is_anomaly"].astype(np.int8).to_numpy() if "is_anomaly" in d else -1
        yield _seri(f"wind_gearbox/{f}", "wind_gearbox", "energy", _unix(d["timestamp"]), d[names].to_numpy(),
                    lab, names, "row", False,
                    note="dişli kutusu SCADA; labeled: 3 olay, complex: etiketsiz (anomali içerebilir)")


def load_hai():
    """HAI (HIL-based Augmented ICS, NSR): kazan/türbin/su arıtma test düzeneği, 1 sn SCADA.
    Dört sürüm; train dosyaları saldırısız (0), test dosyaları saldırı etiketli.
    attack_P1..P3 varsa etiket ilgili prosesin sütunlarına yazılır (cell), yoksa satır etiketi."""
    base = RAW / "hai"
    for ver in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for f in sorted((base / ver).glob("*.csv")):
            if f.name.startswith("label-"):
                continue
            sep = ";" if ";" in f.open().readline() else ","
            d = pd.read_csv(f, sep=sep)
            d.columns = d.columns.str.strip()
            tcol = d.columns[0]
            labcols = [c for c in d.columns if c.lower().startswith("attack")]
            if not labcols and (f.parent / f.name.replace("hai-", "label-")).exists() and "test" in f.name:
                lab_df = pd.read_csv(f.parent / f.name.replace("hai-", "label-"))
                d = d.merge(lab_df, left_on=tcol, right_on="timestamp", how="left", suffixes=("", "_lab"))
                d["attack"] = d["label"].fillna(0)
                labcols = ["attack"]
                d = d.drop(columns=[c for c in ("label", "timestamp_lab") if c in d])
            names = [c for c in d.columns if c not in labcols and c != tcol and pd.api.types.is_numeric_dtype(d[c])]
            names = [c for c in names if d[c].std() > 0]                 # sabit sütunlar atılır
            X = d[names].to_numpy()
            T, k = X.shape
            split = "train" if "train" in f.name else "test"
            if split == "train":
                lab, level = np.zeros((T, k), dtype=np.int8), "row"
            else:
                row = d[labcols[0]].fillna(0).astype(np.int8).to_numpy()
                per_p = {c[-2:]: d[c].fillna(0).astype(np.int8).to_numpy() for c in labcols if "_P" in c}
                if per_p:
                    lab = np.zeros((T, k), dtype=np.int8)
                    covered = np.zeros(T, dtype=bool)
                    for pfx, v in per_p.items():
                        cols = [j for j, n in enumerate(names) if n.startswith(pfx + "_")]
                        lab[np.ix_(v == 1, cols)] = 1
                        covered |= v == 1
                    lab[(row == 1) & ~covered] = 1                       # proses bilgisi olmayan saldırı: tüm sütunlar
                    level = "cell"
                else:
                    lab, level = row, "row"
            yield _seri(f"hai/{ver}/{f.stem.replace('hai-', '')}", "hai", "process_control", _unix(d[tcol]), X,
                        lab, names, level, False,
                        note=f"ICS test düzeneği (kazan/türbin/su); {ver}; train saldırısız")


# -----------------------------------------------------------------------------
# LOTSA (Salesforce/lotsa_data): etiketsiz tahmin derlemi, "normal" arka plan için.
# Alt küme başına en küçük Arrow dosyası indirilir; seri ve satır sayısı sınırlandırılır.
# -----------------------------------------------------------------------------
LOTSA_REPO = "Salesforce/lotsa_data"
LOTSA_CACHE = RAW / "hf_cache"
LOTSA_MAX_SERIES = 200        # alt küme başına seri
LOTSA_MAX_ROWS = 10_000       # seri başına satır (rastgele bitişik parça)
LOTSA_MIN_ROWS = 64
LOTSA_SKIP = re.compile(r"^(era5_(?!2018)|cmip6_(?!2010)|largest_(?!2021))")   # yıl kopyalarından birer tane

LOTSA_SECTOR = [   # (regex, sektör) — ilk eşleşen kazanır
    (r"^(BEIJING_SUBWAY|HZMETRO|SHMETRO|LOOP_SEATTLE|LOS_LOOP|M_DENSE|PEMS|Q-TRAFFIC|SZ_TAXI|taxi_|uber_tlc|"
     r"traffic_|pedestrian|rideshare|vehicle_trips|largest_|covid_mobility)", "transport"),
    (r"^(australian_electricity|bdg-2|buildings_900k|covid19_energy|elecdemand|elf|gfc1|ideal|lcl|london_smart|"
     r"residential_|solar_power|wind_power|wind_farms|spain|smart|sceaux|borealis|bull|cockatoo|hog|pdb|kdd2022)", "energy"),
    (r"^(era5|cmip6|weather|oikolab|subseasonal|temperature_rain|china_air|beijing_air|kdd_cup_2018|saugeenday|sunspot)",
     "environment"),
    (r"^(borg_cluster|azure_vm|alibaba_cluster|extended_web_traffic|kaggle_web_traffic|wiki-rolling)", "it"),
    (r"^(favorita|m5|restaurant|hierarchical_sales|car_parts|godaddy)", "retail"),
    (r"^(bitcoin|fred_md|nn5)", "finance"),
    (r"^(hospital|covid_deaths|cdc_fluview|project_tycho|us_births)", "health"),
    (r"^tourism", "tourism"),
    (r"^(m1_|m3_|m4_|monash_m3|cif_2016)", "mixed_business"),
]


def _lotsa_sector(name):
    return next((sec for pat, sec in LOTSA_SECTOR if re.match(pat, name)), "unknown")


def lotsa_plan():
    """İndirilecek (alt küme, dosya) listesi: her alt kümenin en küçük Arrow dosyası."""
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(LOTSA_REPO, files_metadata=True)
    best = {}
    for f in info.siblings:
        parts = f.rfilename.split("/")
        if len(parts) != 2 or not parts[1].endswith(".arrow") or LOTSA_SKIP.match(parts[0]):
            continue
        if parts[0] not in best or (f.size or 0) < best[parts[0]][1]:
            best[parts[0]] = (f.rfilename, f.size or 0)
    return {k: v[0] for k, v in sorted(best.items())}


def lotsa_indir(plan):
    from huggingface_hub import hf_hub_download
    return {k: hf_hub_download(LOTSA_REPO, f, repo_type="dataset", cache_dir=LOTSA_CACHE) for k, f in plan.items()}


def _lotsa_time(start, freq, T):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = pd.date_range(pd.Timestamp(start), periods=T, freq=freq)
        return idx.astype("int64").to_numpy() / 1e9, False
    except Exception:
        try:
            step = pd.tseries.frequencies.to_offset(freq).nanos / 1e9
        except Exception:
            step = 3600.0
        return _synthetic_time(T, step), True


def load_lotsa(subset, path):
    import pyarrow.ipc as ipc
    rng = np.random.default_rng(abs(hash(subset)) % 2**32)
    with open(path, "rb") as fh:
        magic = fh.read(6)
        fh.seek(0)
        tbl = ipc.open_file(fh).read_all() if magic == b"ARROW1" else ipc.open_stream(fh).read_all()
    n = tbl.num_rows
    pick = np.sort(rng.choice(n, min(n, LOTSA_MAX_SERIES), replace=False))
    for i in pick:
        r = tbl.slice(int(i), 1).to_pylist()[0]
        tg = r["target"]
        X = np.asarray(tg, dtype=np.float32).T if isinstance(tg[0], list) else np.asarray(tg, dtype=np.float32)[:, None]
        T = len(X)
        if T < LOTSA_MIN_ROWS or np.isnan(X).all():
            continue
        t, synth = _lotsa_time(r["start"], r["freq"], T)
        if T > LOTSA_MAX_ROWS:                      # rastgele bitişik parça
            a = int(rng.integers(0, T - LOTSA_MAX_ROWS + 1))
            X, t = X[a:a + LOTSA_MAX_ROWS], t[a:a + LOTSA_MAX_ROWS]
        k = X.shape[1]
        yield _seri(f"lotsa/{subset}/{r['item_id']}", "lotsa", _lotsa_sector(subset), t, X, -1,
                    [f"dim_{j}" for j in range(k)], "row", synth, note=f"etiketsiz; freq={r['freq']}")


LOADERS = {"skab": load_skab, "skab_teaser": load_skab_teaser, "nab": load_nab, "smap_msl": load_smap_msl,
           "pump": load_pump, "smd": load_smd, "cnc": load_cnc, "wind_gearbox": load_wind_gearbox,
           "hai": load_hai}


# =============================================================================
# Okuma
# =============================================================================
def seri_yukle(series_id):
    """Havuzdan bir seriyi kullanıcı biçiminde döndürür."""
    kat = pd.read_csv(KATALOG).set_index("series_id").loc[series_id]
    d = pd.read_parquet(OUT / kat.file, filters=[("series_id", "==", series_id)])
    T, k = int(kat.n_rows), int(kat.n_channels)
    t = d["timestamp"].to_numpy()[::k]
    X = d["value"].to_numpy(np.float64).reshape(T, k)
    labels = d["label"].to_numpy().reshape(T, k)
    return {"raw": np.column_stack([t, X]), "labels": labels, "meta": kat.to_dict()}


def _yaz(name, gen, kat_rows):
    parts, kats = [], []
    for long, kat in gen:
        parts.append(long)
        kat["file"] = f"{name}.parquet"
        kats.append(kat)
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df["series_id"] = df["series_id"].astype("category")
    (OUT / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet", index=False, compression="zstd")
    kat_rows += kats
    n_rows = sum(k["n_rows"] for k in kats)
    print(f"{name:40s} {len(kats):5d} seri  {n_rows:>11,d} satır  {len(df):>13,d} hücre", flush=True)


def main(sources):
    OUT.mkdir(parents=True, exist_ok=True)
    kat_rows = []
    old = pd.read_csv(KATALOG) if KATALOG.exists() else None
    for name in sources:
        if name == "lotsa":
            plan = lotsa_plan()
            print(f"lotsa: {len(plan)} alt küme indiriliyor...", flush=True)
            for sub, path in lotsa_indir(plan).items():
                _yaz(f"lotsa/{sub}", load_lotsa(sub, path), kat_rows)
            continue
        _yaz(name, LOADERS[name](), kat_rows)
    new = pd.DataFrame(kat_rows)
    if old is not None:
        old = old[~old["series_id"].isin(new["series_id"])]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values("series_id").to_csv(KATALOG, index=False)
    print(f"katalog: {KATALOG} ({len(new)} seri)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(LOADERS))
