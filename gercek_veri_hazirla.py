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
             "skab": "SKAB benchmark", "ucr": "UCR Anomaly Archive", "psm": "TSB-AD-M"}

LISANS = {"skab": "AGPL-3.0", "skab_teaser": "AGPL-3.0", "nab": "AGPL-3.0",
          "smap": "telemanom (Apache-2.0), veri NASA", "msl": "telemanom (Apache-2.0), veri NASA",
          "pump": "belirsiz (Kaggle: unknown)", "smd": "MIT", "cnc": "CC0-1.0",
          "wind_gearbox": "Apache-2.0", "hai": "CC-BY-SA-4.0", "metropt": "CC-BY-4.0",
          "cats": "CC-BY-4.0", "tep": "CC-BY-4.0 (Rieth vd. 2017, simülasyon)",
          "cmapss": "CC0 / NASA kamu malı", "hydraulic": "CC-BY-4.0 (UCI)", "telecom_milan": "CC-BY (Telecom Italia)",
          "bidmc": "ODC-BY 1.0 (PhysioNet)", "mitbih": "ODC-BY 1.0 (PhysioNet)", "batadal": "belirsiz (BATADAL yarışması)",
          "ved": "Apache-2.0", "glucobench": "belirsiz (Kaggle)", "stocks": "CC0-1.0",
          "esa": "CC-BY-4.0 (ESA-ADB)", "ppg_dalia": "CC-BY-4.0 (UCI)", "bosch_cnc": "CC-BY-4.0",
          "lbnl": "CC-BY-4.0", "ims_bearing": "NASA kamu malı", "loghub": "belirsiz (Loghub)", "binance": "belirsiz (Kaggle)",
          "ucr": "akademik kullanım (UCR)", "psm": "eBay (RANSynCoders, MIT)", "damadics": "akademik (DAMADICS, Lublin)",
          "asd": "InterFusion (MIT)", "uci": "CC-BY-4.0 (UCI)", "femto": "PHM 2012 / FEMTO-ST (akademik)"}


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


# Veloso vd. 2022, "The MetroPT dataset for predictive maintenance", Tablo: hava kompresörü arızaları
METROPT_FAILURES = [("2020-04-18 00:00", "2020-04-18 23:59"), ("2020-05-29 23:30", "2020-05-30 06:00"),
                    ("2020-06-05 10:00", "2020-06-07 14:30"), ("2020-07-15 14:30", "2020-07-15 19:00")]


def load_metropt():
    """MetroPT-3 (Porto metrosu APU kompresörü): 10 sn, 15 sensör (7 analog + 8 dijital), 4 arıza."""
    d = pd.read_csv(RAW / "metropt" / "MetroPT3(AirCompressor).csv", index_col=0)
    t = _unix(d["timestamp"])
    names = [c for c in d.columns if c != "timestamp"]
    lab = np.zeros(len(d), dtype=np.int8)
    for a, b in METROPT_FAILURES:
        lab[(t >= _unix(pd.Series([a]))[0]) & (t <= _unix(pd.Series([b]))[0])] = 1
    yield _seri("metropt/0", "metropt", "transport", t, d[names].to_numpy(), lab, names, "row", False,
                note="etiket = makaledeki 4 arıza aralığı (yağ sızıntısı / hava kaçağı)")


def load_cats():
    """CATS (Solenix): 17 kanal, 1 Hz, 200 kontrollü anomali; kök neden + etkilenen kanallar metadata'da.
    İlk %70 eğitim (train), kalan %30 doğrulama (val)."""
    base = RAW / "cats"
    d = pd.read_csv(base / "data.csv")
    meta = pd.read_csv(base / "metadata.csv")
    t = _unix(d["timestamp"])
    names = [c for c in d.columns if c not in ("timestamp", "y", "category")]
    X = d[names].to_numpy()
    T, k = X.shape
    lab = np.zeros((T, k), dtype=np.int8)
    for _, r in meta.iterrows():
        rows = (t >= _unix(pd.Series([r.start_time]))[0]) & (t <= _unix(pd.Series([r.end_time]))[0])
        cols = [names.index(c) for c in [r.root_cause] + ast.literal_eval(r.affected) if c in names]
        lab[np.ix_(rows, cols)] = 1
    y = d["y"].to_numpy() > 0
    lab[y & ~(lab == 1).any(1)] = 1                                   # metadata dışı etiketli satır: tüm sütunlar
    cut = int(T * 0.7)
    for split, sl in (("train", slice(0, cut)), ("val", slice(cut, T))):
        yield _seri(f"cats/{split}", "cats", "space", t[sl], X[sl], lab[sl], names, "cell", False,
                    note="kontrollü anomali (uydu benzeri simülasyon test düzeneği); kök neden + etkilenen kanal etiketli")


def load_tep(runs_per_fault=25, normal_runs=50):
    """Tennessee Eastman (Rieth 2017): 52 değişken, 3 dk, koşu başına 500 örnek; arıza 20. örnekte başlar."""
    base = RAW / "tep"
    rng = np.random.default_rng(0)
    names = None
    for f, faulty in (("TEP_FaultFree_Training.csv", False), ("TEP_Faulty_Training.csv", True)):
        d = pd.read_csv(base / f, index_col=0)
        names = names or [c for c in d.columns if c.startswith(("xmeas", "xmv"))]
        for fault, g in d.groupby("faultNumber"):
            runs = sorted(g.simulationRun.unique())
            pick = rng.choice(runs, min(len(runs), runs_per_fault if faulty else normal_runs), replace=False)
            for run in pick:
                r = g[g.simulationRun == run].sort_values("sample")
                T = len(r)
                lab = np.zeros(T, dtype=np.int8)
                if faulty:
                    lab[r["sample"].to_numpy() > 20] = 1
                yield _seri(f"tep/fault{int(fault):02d}/run{int(run)}", "tep", "process_control", _synthetic_time(T, 180.0),
                            r[names].to_numpy(), lab, names, "row", True,
                            note=f"simülasyon; arıza {int(fault)} 20. örnekten sonra (0 = arızasız)")


def load_cmapss(last_cycles=25):
    """NASA C-MAPSS turbofan: motor başına çalışma-arıza koşusu, 21 sensör + 3 ayar, 1 satır = 1 uçuş.
    Etiket: son `last_cycles` çevrim (arızaya yaklaşan bozulma) = 1. Zaman sentetik (1 çevrim = 1 saat)."""
    base = RAW / "cmapss" / "CMaps"
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        d = pd.read_csv(base / f"train_{fd}.txt", sep=r"\s+", header=None, names=cols)
        for unit, g in d.groupby("unit"):
            g = g.sort_values("cycle")
            X = g[cols[2:]].to_numpy()
            X = X[:, X.std(0) > 0]
            names = [c for c, ok in zip(cols[2:], d.loc[g.index, cols[2:]].std(0) > 0) if ok]
            T = len(g)
            lab = np.zeros(T, dtype=np.int8)
            lab[-last_cycles:] = 1
            yield _seri(f"cmapss/{fd}/unit{int(unit)}", "cmapss", "manufacturing", _synthetic_time(T, 3600.0), X, lab,
                        names, "row", True, note=f"turbofan çalışma-arıza; son {last_cycles} çevrim etiketli")


def load_hydraulic():
    """UCI hidrolik test düzeneği: 2205 çevrim × 60 sn, 17 sensör (100/10/1 Hz). Hepsi 1 Hz'e indirilip
    çevrimler art arda eklenir. Koşullar deney tasarımı (DOE) ile eşit dağıtıldığı için anomali etiketi
    verilmez (-1); imalat arka planı olarak kullanılır."""
    base = RAW / "hydraulic"
    names = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
    cols = []
    for n in names:
        a = np.loadtxt(base / f"{n}.txt")                  # (2205, 60·Hz)
        f = a.shape[1] // 60
        cols.append(a.reshape(a.shape[0], 60, f).mean(2).reshape(-1))
    X = np.column_stack(cols)
    yield _seri("hydraulic/0", "hydraulic", "manufacturing", _synthetic_time(len(X), 1.0), X, -1, names, "row", True,
                note="çevrimler art arda; koşullar (soğutucu/valf/kaçak/akümülatör) profile.txt'de, DOE → etiketsiz")


def load_telecom_milan(n_cells=400):
    """Telecom Italia Milano (Kaggle sürümü): saatlik, hücre bazında SMS/çağrı/internet, 7 gün.
    Ülke kodları toplanır; en yoğun n_cells hücre alınır. Etiketsiz telekom arka planı."""
    base = RAW / "telecom_milan"
    files = sorted(base.glob("sms-call-internet-mi-*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ch = ["smsin", "smsout", "callin", "callout", "internet"]
    agg = d.groupby(["CellID", "datetime"])[ch].sum(min_count=1).reset_index()
    top = agg.groupby("CellID")[ch].count().sum(1).sort_values(ascending=False).index[:n_cells]
    for cell in top:
        g = agg[agg.CellID == cell].sort_values("datetime")
        yield _seri(f"telecom_milan/cell{int(cell)}", "telecom_milan", "telecom", _unix(g["datetime"]),
                    g[ch].to_numpy(), -1, ch, "row", False, note="etiketsiz; saatlik hücre trafiği (1 hafta)")


def load_bidmc():
    """BIDMC (PhysioNet): 53 yoğun bakım hastası, 8 dk. Signals 125 Hz (RESP, PLETH, EKG V/AVR/II),
    Numerics 1 Hz (HR, PULSE, RESP, SpO2). Etiketsiz sağlık arka planı."""
    base = next(RAW.glob("bidmc/*/bidmc_csv"))
    for f in sorted(base.glob("bidmc_*_Signals.csv")):
        rid = f.name.split("_")[1]
        sig = pd.read_csv(f); sig.columns = sig.columns.str.strip()
        names = [c for c in sig.columns if c != "Time [s]"]
        yield _seri(f"bidmc/{rid}/signals", "bidmc", "ecg", sig["Time [s]"].to_numpy(dtype=float), sig[names].to_numpy(),
                    -1, names, "row", True, note="125 Hz dalga formu (EKG, PPG, solunum); zaman kayıt başından itibaren")
        num = pd.read_csv(f.with_name(f"bidmc_{rid}_Numerics.csv")); num.columns = num.columns.str.strip()
        names = [c for c in num.columns if c != "Time [s]"]
        yield _seri(f"bidmc/{rid}/numerics", "bidmc", "vitals", num["Time [s]"].to_numpy(dtype=float), num[names].to_numpy(),
                    -1, names, "row", True, note="1 Hz vital bulgular")


def load_batadal():
    """BATADAL: su dağıtım şebekesi (C-Town), saatlik, 43 sensör (tank seviyesi, pompa akış/durum, basınç).
    training_1 saldırısız; training_2 ve test saldırı etiketli (ATT_FLAG)."""
    base = RAW / "batadal"
    for f in ("training_dataset_1", "training_dataset_2", "test_dataset"):
        d = pd.read_csv(base / f"{f}.csv"); d.columns = d.columns.str.strip()
        names = [c for c in d.columns if c not in ("DATETIME", "ATT_FLAG")]
        t = pd.to_datetime(d["DATETIME"], format="%d/%m/%y %H").astype("int64").to_numpy() / 1e9
        lab = (d["ATT_FLAG"].fillna(0).to_numpy() > 0).astype(np.int8)
        yield _seri(f"batadal/{f}", "batadal", "water", t, d[names].to_numpy(), lab, names, "row", False,
                    note="siber saldırı etiketi (ATT_FLAG); training_1 tamamen normal")


NORMAL_BEATS = {"N", "L", "R", "e", "j", ".", "+", "~", "|", "\"", "x", "s", "@", "[", "]", "!", "(", ")", "p", "t", "u", "`", "'", "^"}


def load_mitbih(halfwin=54):
    """MIT-BIH Arrhythmia (PhysioNet, CSV): 48 kayıt × 30 dk, 360 Hz, 2 derivasyon.
    Etiket: normal olmayan atımlar (V, A, F, /, ...) ± halfwin örnek (~0.15 sn)."""
    base = RAW / "mitbih"
    for f in sorted(base.glob("*_ekg.csv")):
        rid = f.name.split("_")[0]
        d = pd.read_csv(f, index_col=0)
        names = [c for c in d.columns if c != "symbol"]
        X = d[names].to_numpy()
        T = len(X)
        lab = np.zeros(T, dtype=np.int8)
        beats = d.index[d["symbol"].notna() & ~d["symbol"].isin(NORMAL_BEATS)].to_numpy()
        for b in beats:
            lab[max(0, b - halfwin):b + halfwin] = 1
        yield _seri(f"mitbih/{rid}", "mitbih", "ecg", _synthetic_time(T, 1 / 360), X, lab, names, "row", True,
                    note="anormal atım (ektopik vb.) etiketi; 360 Hz")


def load_ved(max_trips=500, min_rows=200):
    """VED (Vehicle Energy Dataset, Michigan): yolculuk başına ~1 sn CAN/OBD: hız, MAF, devir, yük, dış sıcaklık,
    yakıt düzeltmeleri. Etiketsiz otomotiv arka planı."""
    base = RAW / "ved"
    ch = ["Vehicle_Speed_km_per_h", "MAF_g_per_sec", "Engine_RPM_RPM", "Absolute_Load_pct", "OAT_DegC",
          "Short_Term_Fuel_Trim_Bank_1_pct", "Long_Term_Fuel_Trim_Bank_1_pct"]
    rng = np.random.default_rng(0)
    n = 0
    for f in sorted(base.rglob("*.parquet")):
        d = pd.read_parquet(f, columns=["VehId", "Trip", "Timestampms"] + ch)
        trips = [(v, tr) for (v, tr), g in d.groupby(["VehId", "Trip"]).size().items() if g >= min_rows]
        rng.shuffle(trips)
        for v, tr in trips:
            if n >= max_trips:
                return
            g = d[(d.VehId == v) & (d.Trip == tr)].sort_values("Timestampms")
            X = g[ch].to_numpy(dtype=float)
            keep = ~np.isnan(X).all(0)
            if keep.sum() < 3:
                continue
            t0 = 1_527_000_000.0 + n * 1e5                       # yolculuklar birbirinden ayrık sahte zamanlar
            yield _seri(f"ved/veh{int(v)}/trip{int(tr)}", "ved", "automotive", t0 + g["Timestampms"].to_numpy() / 1000.0,
                        X[:, keep], -1, [c for c, k in zip(ch, keep) if k], "row", True,
                        note="etiketsiz; yolculuk içi göreli zaman gerçek, başlangıç sahte")
            n += 1


def load_stocks(n_stocks=400, n_etfs=100, min_rows=1000):
    """Huge Stock Market Dataset (CC0): ABD hisse/ETF günlük OHLCV. Rastgele örneklem, iş günü takvimi.
    Etiketsiz finans arka planı (hafta sonu boşlukları normal)."""
    base = RAW / "stocks"
    rng = np.random.default_rng(0)
    ch = ["Open", "High", "Low", "Close", "Volume"]
    for sub, n in (("Stocks", n_stocks), ("ETFs", n_etfs)):
        files = sorted((base / sub).glob("*.txt"))
        for f in rng.choice(files, min(n, len(files)), replace=False):
            try:
                d = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if len(d) < min_rows:
                continue
            yield _seri(f"stocks/{sub.lower()}/{f.stem.replace('.us', '')}", "stocks", "finance", _unix(d["Date"]),
                        d[ch].to_numpy(dtype=float), -1, ch, "row", False, note="etiketsiz; günlük OHLCV, iş günleri")


def load_esa(resample="10min"):
    """ESA-ADB Mission1 (ESA, 2024): gerçek uydu telemetrisi, 14 yıl, düzensiz örnekleme; kanal bazında
    etiketli anomali aralıkları. İndirilen kanallar 10 dk ortalamaya indirilir; ilk %70 train, kalan val."""
    base = RAW / "esa"
    labels = pd.read_csv(base / "labels.csv")
    labels["StartTime"] = pd.to_datetime(labels.StartTime).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels.EndTime).dt.tz_localize(None)
    for z in sorted(base.glob("channel_*.zip"), key=lambda p: int(p.stem.split("_")[1])):
        ch = z.stem
        f = base / ch
        if not f.exists():
            import zipfile
            zipfile.ZipFile(z).extractall(base)
        d = pd.read_pickle(f)
        r = d.iloc[:, 0].resample(resample).mean()
        r = r[r.first_valid_index():r.last_valid_index()]
        t = r.index.astype("int64").to_numpy() / 1e9
        lab = np.zeros(len(r), dtype=np.int8)
        for _, row in labels[labels.Channel == ch].iterrows():
            lab[(r.index >= row.StartTime) & (r.index <= row.EndTime)] = 1
        cut = int(len(r) * 0.7)
        for split, sl in (("train", slice(0, cut)), ("val", slice(cut, None))):
            yield _seri(f"esa/{ch}/{split}", "esa", "space", t[sl], r.to_numpy()[sl, None], lab[sl], [ch], "row", False,
                        note=f"ESA-ADB Mission1, {resample} ortalama; anomali aralıkları labels.csv")


def load_bosch_cnc(max_files_per_op=40, down=10):
    """Bosch CNC Machining: 3 makine × 15 operasyon, 2 kHz 3 eksen titreşim; dosya = 1 proses (iyi/kötü).
    Aynı (makine, op) dosyaları zaman sırasıyla art arda eklenir, kötü prosesler etiketlenir; 10× ortalama (200 Hz)."""
    base = RAW / "bosch_cnc"
    for m in sorted(p for p in base.iterdir() if p.is_dir()):
        for op in sorted(p for p in m.iterdir() if p.is_dir()):
            files = sorted(op.rglob("*.csv"), key=lambda p: p.name)[:max_files_per_op]
            if not files:
                continue
            parts, labs = [], []
            for f in files:
                a = pd.read_csv(f).to_numpy(dtype=float)
                n = len(a) // down
                a = a[:n * down].reshape(n, down, -1).mean(1)
                parts.append(a); labs.append(np.full(n, 1 if f.parent.name == "bad" else 0, dtype=np.int8))
            X, lab = np.vstack(parts), np.concatenate(labs)
            yield _seri(f"bosch_cnc/{m.name}/{op.name}", "bosch_cnc", "manufacturing", _synthetic_time(len(X), down / 2000),
                        X, lab, ["acc_x", "acc_y", "acc_z"], "row", True,
                        note=f"{len(files)} proses art arda; kötü proses = 1")


def load_lbnl():
    """LBNL bina HVAC arıza tespiti: 1 dk, AHU/RTU/VAV sensörleri, 'Fault Detection Ground Truth' etiketi.
    Tamamı arızalı olan dosyalar etiketsiz (-1) sayılır."""
    base = RAW / "lbnl"
    for f in sorted(base.glob("*.csv")):
        d = pd.read_csv(f, na_values=["NA"])
        lcol = "Fault Detection Ground Truth"
        tcol = d.columns[0]
        names = [c for c in d.columns if c not in (tcol, lcol) and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().any()]
        y = d[lcol].fillna(0).astype(int).to_numpy()
        lab = y.astype(np.int8) if 0 < y.mean() < 1 else -1
        yield _seri(f"lbnl/{f.stem}", "lbnl", "building", _unix(pd.to_datetime(d[tcol], format="mixed")),
                    d[names].to_numpy(dtype=float), lab, names, "row", False,
                    note="HVAC arıza senaryoları; dosya tamamen arızalıysa etiketsiz")


def load_ims_bearing(raw_snapshots=30):
    """NASA IMS rulman: her 10 dk'da 1 sn (20 kHz) titreşim anlık kaydı, arızaya kadar.
    (a) anlık kayıt başına RMS / tepe / basıklık → uzun bozulma serisi (son %8 = 1)
    (b) rastgele anlık kayıtlar 20 kHz ham arka plan olarak."""
    base = RAW / "ims_bearing"
    rng = np.random.default_rng(0)
    for test in ("1st_test", "2nd_test", "3rd_test"):
        files = sorted(p for p in (base / test).rglob("*") if p.is_file() and p.name[:4].isdigit())
        feats, times = [], []
        for f in files:
            a = np.loadtxt(f)
            rms = np.sqrt((a ** 2).mean(0)); peak = np.abs(a).max(0)
            kurt = ((a - a.mean(0)) ** 4).mean(0) / (a.var(0) ** 2 + 1e-12)
            feats.append(np.concatenate([rms, peak, kurt]))
            times.append(pd.to_datetime(f.name, format="%Y.%m.%d.%H.%M.%S").timestamp())
        F = np.array(feats); k = F.shape[1] // 3
        names = [f"{s}_b{i+1}" for s in ("rms", "peak", "kurt") for i in range(k)]
        lab = np.zeros(len(F), dtype=np.int8); lab[int(len(F) * 0.92):] = 1
        yield _seri(f"ims_bearing/{test}/features", "ims_bearing", "manufacturing", np.array(times), F, lab, names, "row", False,
                    note="anlık kayıt özellikleri (10 dk); son %8 = arızaya yaklaşma")
        for f in rng.choice(files, min(raw_snapshots, len(files)), replace=False):
            a = np.loadtxt(f)
            yield _seri(f"ims_bearing/{test}/raw_{f.name}", "ims_bearing", "manufacturing", _synthetic_time(len(a), 1 / 20480), a, -1,
                        [f"acc_{i+1}" for i in range(a.shape[1])], "row", True, note="1 sn ham titreşim, 20 kHz")


def load_bgl(bin_s=60):
    """Loghub BGL (BlueGene/L) sistem logu: dakikalık mesaj sayaçları; satır başındaki '-' = normal,
    diğer etiketler = alarm. Kanallar: toplam, KERNEL, APP, HARDWARE, diğer; etiket = dakikada alarm var mı."""
    base = RAW / "loghub"
    ts, alert, comp = [], [], []
    with open(base / "BGL.log", errors="ignore") as fh:
        for line in fh:
            parts = line.split(" ", 9)
            if len(parts) < 9:
                continue
            ts.append(int(parts[1])); alert.append(parts[0] != "-"); comp.append(parts[7])
    d = pd.DataFrame({"t": np.array(ts) // bin_s * bin_s, "alert": alert, "comp": comp})
    g = d.groupby("t")
    X = pd.DataFrame({"total": g.size(), "kernel": g.comp.apply(lambda c: (c == "KERNEL").sum()),
                      "app": g.comp.apply(lambda c: (c == "APP").sum()), "hardware": g.comp.apply(lambda c: (c == "HARDWARE").sum())})
    lab = g.alert.any().astype(np.int8).reindex(X.index).to_numpy()
    full = pd.RangeIndex(X.index.min(), X.index.max() + bin_s, bin_s)
    X = X.reindex(full, fill_value=0); lab = pd.Series(lab, index=g.size().index).reindex(full, fill_value=0).to_numpy()
    yield _seri("bgl/0", "loghub", "it", X.index.to_numpy(dtype=float), X.to_numpy(dtype=float), lab, list(X.columns), "row", False,
                note="dakikalık log sayaçları; alarm etiketli dakikalar = 1")


def load_binance():
    """Binance 1 dk OHLCV (Kaggle): BTC/ETH/BNB/ADA-USDT. Etiketsiz kripto arka planı (7/24, boşluksuz)."""
    base = RAW / "binance"
    ch = ["open", "high", "low", "close", "volume", "number_of_trades"]
    for f in sorted(base.glob("*.parquet")):
        d = pd.read_parquet(f, columns=ch).dropna()
        yield _seri(f"binance/{f.stem}", "binance", "finance", d.index.astype("int64").to_numpy() / 1e9,
                    d.to_numpy(dtype=float), -1, ch, "row", False, note="etiketsiz; 1 dk OHLCV")


def load_ucr():
    """UCR Time Series Anomaly Archive (2021): 250 tek değişkenli seri; dosya adı = ..._TRAINEND_ASTART_AEND.
    SADECE DEĞERLENDİRME (README §9). Test bölümü (TRAINEND sonrası) etiketli; eğitim bölümü normal."""
    base = RAW / "ucr"
    seen = set()
    for f in sorted(base.rglob("*.txt")):
        parts = f.stem.split("_")
        if len(parts) < 4 or not parts[-1].isdigit() or f.stem in seen:     # arşivde birkaç dosya iki kopya
            continue
        seen.add(f.stem)
        try:
            x = np.loadtxt(f)
        except ValueError:
            x = np.loadtxt(f, delimiter=",").ravel()
        x = x.ravel()
        train_end, a, b = int(parts[-3]), int(parts[-2]), int(parts[-1])
        lab = np.zeros(len(x), dtype=np.int8)
        lab[a:b + 1] = 1
        name = "_".join(parts[:-3])
        yield _seri(f"ucr/{name}", "ucr", "abstract", _synthetic_time(len(x), 1.0), x[:, None], lab, ["value"], "row", True,
                    note=f"UCR anomali arşivi; eğitim bölümü ilk {train_end} nokta, anomali [{a},{b}]")


def load_psm():
    """PSM (eBay Pooled Server Metrics): 25 metrik, 1 dk; test etiketli. SADECE DEĞERLENDİRME (TSB-AD-M)."""
    base = RAW / "psm"
    for split in ("train", "test"):
        d = pd.read_csv(base / f"{split}.csv")
        names = [c for c in d.columns if c.startswith("feature")]
        X = d[names].to_numpy(dtype=float)
        if split == "test":
            lab = pd.read_csv(base / "test_label.csv")["label"].to_numpy().astype(np.int8)
        else:
            lab = -1
        yield _seri(f"psm/{split}", "psm", "it", _synthetic_time(len(X), 60.0), X, lab, names, "row", True,
                    note="eBay sunucu metrikleri; train etiketsiz, test etiketli")


def load_damadics():
    """DAMADICS (Lublin şeker fabrikası, Kasım 2001): aktüatör/proses ölçümleri, 1 Hz, günlük dosyalar.
    Arıza zamanları resmi kayıtta; burada etiketsiz proses arka planı olarak alınır (-1)."""
    base = RAW / "damadics"
    for f in sorted(base.glob("*.txt"), key=lambda p: (p.stem[4:8], p.stem[2:4], p.stem[:2])):
        d = pd.read_csv(f, sep=None, engine="python", header=None)
        d = d.select_dtypes("number")
        X = d.to_numpy(dtype=float)
        X = X[:, X.std(0) > 0]
        day = pd.Timestamp(f"{f.stem[4:8]}-{f.stem[2:4]}-{f.stem[:2]}").timestamp()
        yield _seri(f"damadics/{f.stem}", "damadics", "process_control", day + np.arange(len(X), dtype=float), X, -1,
                    [f"v{i}" for i in range(X.shape[1])], "row", True, note="etiketsiz; günlük 1 Hz kayıt")


def load_asd():
    """ASD (Application Server Dataset, InterFusion): 12 sunucu, 19 metrik, 5 dk. train etiketsiz (normal), test etiketli."""
    import pickle
    base = RAW / "asd"
    for i in range(1, 13):
        tr = np.asarray(pickle.load(open(base / f"omi-{i}_train.pkl", "rb")), dtype=float)
        te = np.asarray(pickle.load(open(base / f"omi-{i}_test.pkl", "rb")), dtype=float)
        lab = np.asarray(pickle.load(open(base / f"omi-{i}_test_label.pkl", "rb"))).astype(np.int8)
        names = [f"metric_{j}" for j in range(tr.shape[1])]
        yield _seri(f"asd/omi-{i}/train", "asd", "it", _synthetic_time(len(tr), 300.0), tr, -1, names, "row", True, note="etiketsiz (normal)")
        yield _seri(f"asd/omi-{i}/test", "asd", "it", _synthetic_time(len(te), 300.0, 1_577_836_800.0 + 300 * len(tr)), te, lab,
                    names, "row", True, note="etiketli test (train'in devamı)")


def load_uci_small():
    """UCI küçük IoT/çevre setleri: hava kalitesi (saatlik), ev enerji (10 dk), oda doluluk (1 dk). Etiketsiz."""
    base = RAW / "uci"
    d = pd.read_csv(base / "air_quality" / "AirQualityUCI.csv", sep=";", decimal=",").dropna(how="all", axis=1).dropna(subset=["Date"])
    t = pd.to_datetime(d["Date"] + " " + d["Time"], format="%d/%m/%Y %H.%M.%S")
    names = [c for c in d.columns if c not in ("Date", "Time")]
    X = d[names].to_numpy(dtype=float); X[X == -200] = np.nan
    yield _seri("uci/air_quality", "uci", "environment", _unix(t), X, -1, names, "row", False, note="etiketsiz; -200 → NaN")
    d = pd.read_csv(base / "appliances" / "energydata_complete.csv")
    names = [c for c in d.columns if c != "date"]
    yield _seri("uci/appliances", "uci", "building", _unix(d["date"]), d[names].to_numpy(dtype=float), -1, names, "row", False, note="etiketsiz")
    for f in ("datatraining", "datatest", "datatest2"):
        d = pd.read_csv(base / "occupancy" / f"{f}.txt")
        names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
        yield _seri(f"uci/occupancy_{f}", "uci", "building", _unix(d["date"]), d[names].to_numpy(dtype=float), -1, names, "row", False, note="etiketsiz")


def _femto_read(f):
    try:
        return np.loadtxt(f, delimiter=",")
    except ValueError:
        return np.loadtxt(f, delimiter=";")


def load_femto(raw_snapshots=25):
    """FEMTO / PRONOSTIA (PHM 2012): rulman çalışma-arıza; her 10 sn'de 0.1 sn (25.6 kHz) 2 eksen titreşim.
    (a) anlık kayıt özellikleri (RMS/tepe/basıklık) → bozulma serisi (son %8 = 1); (b) ham anlık kayıt örnekleri."""
    base = RAW / "femto" / "Learning_set"
    rng = np.random.default_rng(0)
    for b in sorted(base.iterdir()):
        files = sorted(b.glob("acc_*.csv"))
        if not files:
            continue
        feats = []
        for f in files:
            a = _femto_read(f)[:, 4:6]
            rms = np.sqrt((a ** 2).mean(0)); peak = np.abs(a).max(0)
            kurt = ((a - a.mean(0)) ** 4).mean(0) / (a.var(0) ** 2 + 1e-12)
            feats.append(np.concatenate([rms, peak, kurt]))
        F = np.array(feats)
        lab = np.zeros(len(F), dtype=np.int8); lab[int(len(F) * 0.92):] = 1
        yield _seri(f"femto/{b.name}/features", "femto", "manufacturing", _synthetic_time(len(F), 10.0), F, lab,
                    ["rms_h", "rms_v", "peak_h", "peak_v", "kurt_h", "kurt_v"], "row", True, note="10 sn'lik özellikler; son %8 = arızaya yaklaşma")
        for f in rng.choice(files, min(raw_snapshots, len(files)), replace=False):
            a = _femto_read(f)[:, 4:6]
            yield _seri(f"femto/{b.name}/raw_{f.stem}", "femto", "manufacturing", _synthetic_time(len(a), 1 / 25600), a, -1,
                        ["acc_h", "acc_v"], "row", True, note="0.1 sn ham titreşim, 25.6 kHz")


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
           "hai": load_hai, "metropt": load_metropt, "cats": load_cats, "tep": load_tep,
           "cmapss": load_cmapss, "hydraulic": load_hydraulic,
           "telecom_milan": load_telecom_milan, "bidmc": load_bidmc, "batadal": load_batadal, "mitbih": load_mitbih, "ved": load_ved, "stocks": load_stocks, "esa": load_esa,
           "bosch_cnc": load_bosch_cnc, "lbnl": load_lbnl, "ims_bearing": load_ims_bearing, "loghub": load_bgl,
           "binance": load_binance, "ucr": load_ucr, "psm": load_psm, "damadics": load_damadics,
           "asd": load_asd, "uci": load_uci_small, "femto": load_femto}


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
