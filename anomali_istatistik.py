"""
Gerçek etiketli anomali olaylarının istatistiklerini çıkarır, üreticinin ürettikleriyle karşılaştırır
ve eğitim rolündeki kaynaklardan bir "gerçek anomali bankası" (şablonlar) üretir.

    python anomali_istatistik.py            # tablo + data/havuz/anomali_istatistik.csv + data/havuz/anomali_bankasi.npz

Olay: etiketli satırların bitişik bloğu. Ölçüler (olay öncesi bağlama göre, MAD birimi):
    sure         satır sayısı (ve pencere oranı: sure / 4096)
    sutun_oran   etkilenen sütun / toplam sütun (hücre etiketli kaynaklarda)
    genlik       etkilenen sütunlarda |medyan(olay) - medyan(önce)| / MAD(önce), medyan
    baslangic    genliğin yarısına ulaşmak için geçen satır / sure (0 = ani, ~0.5 = kademeli)
    kalici       olay pencerenin/serinin sonuna kadar sürüyor mu
    donus        olay sonrası 2·sure içinde seviye önceki medyana dönüyor mu (|Δ| < 1 MAD)
    varyans_oran MAD(olay) / MAD(önce): >1 gürültü artışı, <1 sakinleşme
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HAVUZ = ROOT / "data" / "havuz"
sys.path.insert(0, str(ROOT))
from gercek_veri_hazirla import seri_yukle  # noqa: E402

TRAIN_LABELED = {  # eğitim rolündeki etiketli kaynaklar (doğrulama/benchmark ASLA)
    "hai": lambda sid: "/test" in sid and ("hai-20.07" in sid or "hai-21.03" in sid),
    "cats": lambda sid: sid.endswith("/train"),
    "esa": lambda sid: sid.endswith("/train"),
    "batadal": lambda sid: "training" in sid,
}
# Dışlananlar: loghub/BGL (tek satırlık log alarmı, sayaç MAD≈0 → genlik anlamsız), Bosch/kantine/LBNL
# (etiket dosya/bölüm boyu blok; "olay" değil rejim). Bunlar eğitimde etiket olarak kalır, istatistik ve bankaya girmez.
MAX_L = 4096
MAX_EVENTS_PER_SOURCE = 400
CTX = 2.0        # olay öncesi bağlam: CTX × süre (en az 32 satır)


def _mad(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    m = np.median(x)
    return np.median(np.abs(x - m)) * 1.4826 + 1e-9


def events_of(sid, t, X, lab, cell_level):
    rows = (lab == 1).any(1)
    if not rows.any():
        return []
    idx = np.flatnonzero(rows)
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]] + 1
    out = []
    for s, e in zip(starts, ends):
        L = e - s
        if L > MAX_L:
            continue
        pre0 = max(0, s - max(32, int(CTX * L)))
        if s - pre0 < 16:
            continue
        cols = np.flatnonzero((lab[s:e] == 1).any(0)) if cell_level else np.arange(X.shape[1])
        pre = X[pre0:s]
        ev = X[s:e]
        amps, onsets, vr, ret = [], [], [], []
        for c in cols:
            m0, sd0 = np.nanmedian(pre[:, c]), _mad(pre[:, c])
            if not np.isfinite(sd0):
                continue
            d = (ev[:, c] - m0) / sd0
            a = np.nanmedian(np.abs(d))
            amps.append(a)
            if a > 0.5:
                half = np.flatnonzero(np.abs(d) >= a / 2)
                onsets.append((half[0] / L) if len(half) else np.nan)
            vr.append(_mad(ev[:, c]) / sd0)
            post = X[e:min(len(X), e + 2 * L), c]
            if len(post) >= 8:
                ret.append(abs(np.nanmedian(post) - m0) / sd0 < 1.0)
        if not amps:
            continue
        # gerçek anomali: etkilenen sütunlar arasında genliği en yüksek olanlar (satır etiketli kaynakta hepsi değil)
        amps = np.array(amps)
        eff = amps >= 0.5
        out.append(dict(series_id=sid, source=sid.split("/")[0], start=int(s), sure=int(L), sure_oran=L / 4096,
                        sutun_oran=(eff.sum() / X.shape[1]) if cell_level else (eff.sum() / X.shape[1]),
                        n_sutun=int(eff.sum()), genlik=float(np.median(amps[eff])) if eff.any() else float(np.median(amps)),
                        baslangic=float(np.nanmedian(onsets)) if onsets else np.nan,
                        kalici=bool(e >= len(X) - 1), donus=float(np.mean(ret)) if ret else np.nan,
                        varyans_oran=float(np.nanmedian(vr)) if vr else np.nan))
    return out


def real_events(kat):
    ev, bank = [], []
    rng = np.random.default_rng(0)
    for src, ok in TRAIN_LABELED.items():
        rows = kat[(kat.source == src) & (~kat.unlabeled) & (kat.series_id.map(ok))]
        for sid in rows.series_id:
            s = seri_yukle(sid)
            X, lab = s["raw"][:, 1:].astype(np.float64), s["labels"]
            cell = s["meta"]["label_level"] == "cell"
            e = events_of(sid, s["raw"][:, 0], X, lab, cell)
            ev += e
            for r in e:                                     # banka: normalize şablon (önce 2L, olay L, sonra L)
                st, L = r["start"], r["sure"]
                if not (8 <= L <= 2048) or r["genlik"] < 0.5:
                    continue
                pre0, post1 = max(0, st - 2 * L), min(len(X), st + 2 * L)
                if st - pre0 < 16:
                    continue
                cols = np.flatnonzero((lab[st:st + L] == 1).any(0)) if cell else np.arange(X.shape[1])
                seg = X[pre0:post1][:, cols]
                pre = X[pre0:st][:, cols]
                m0 = np.nanmedian(pre, 0); sd0 = np.array([_mad(pre[:, j]) for j in range(pre.shape[1])])
                keep = np.isfinite(sd0) & (sd0 > 1e-8)
                if not keep.any():
                    continue
                z = (seg[:, keep] - m0[keep]) / sd0[keep]
                amp = np.nanmedian(np.abs(z[st - pre0:st - pre0 + L]), 0)
                z = z[:, amp >= 0.5]
                if z.shape[1] == 0 or not np.isfinite(z).all():
                    continue
                bank.append(dict(source=src, series_id=sid, pre=st - pre0, L=L, z=np.clip(z, -30, 30).astype(np.float32)))
    if len(bank) > 3000:
        bank = [bank[i] for i in rng.choice(len(bank), 3000, replace=False)]
    return pd.DataFrame(ev), bank


def synthetic_events(n=600, seed=0):
    import egitim_verisi_uretici as g
    rng = np.random.default_rng(seed)
    ev = []
    for _ in range(n):
        s = g.make_sample(rng, difficulty=0.7, force_anomaly=True)
        T, k = s["meta"]["T"], s["meta"]["k"]
        X, lab = s["raw"][:, 1:], s["labels"][:T, :k]
        ev += events_of("sentetik/" + s["meta"]["domain"], s["raw"][:, 0], X, lab, True)
    return pd.DataFrame(ev)


def summarize(df, name):
    # kaynak dengeli: her kaynaktan en fazla 150 olay
    df = df.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=0))
    q = lambda c: df[c].quantile([0.1, 0.5, 0.9]).round(2).tolist()
    return pd.Series({
        "olay": len(df), "sure p10/50/90": q("sure"), "sure/4096 p50": round(df.sure_oran.median(), 3),
        "sutun_oran p50": round(df.sutun_oran.median(), 2), "genlik p10/50/90 (MAD)": q("genlik"),
        "baslangic p50": round(df.baslangic.median(), 2), "kalici %": round(100 * df.kalici.mean(), 1),
        "donus %": round(100 * df.donus.mean(), 1), "varyans_oran p50": round(df.varyans_oran.median(), 2),
        "sakinlesme % (vr<0.5)": round(100 * (df.varyans_oran < 0.5).mean(), 1),
    }, name=name)


if __name__ == "__main__":
    kat = pd.read_csv(HAVUZ / "katalog.csv")
    real, bank = real_events(kat)
    syn = synthetic_events()
    real.to_csv(HAVUZ / "anomali_istatistik.csv", index=False)
    np.savez_compressed(HAVUZ / "anomali_bankasi.npz",
                        source=np.array([b["source"] for b in bank]), series_id=np.array([b["series_id"] for b in bank]),
                        pre=np.array([b["pre"] for b in bank]), L=np.array([b["L"] for b in bank]),
                        z=np.array([b["z"] for b in bank], dtype=object))
    pd.set_option("display.width", 200)
    print(pd.concat([summarize(real, "GERÇEK (eğitim rolü)"), summarize(syn, "SENTETİK (üretici)")], axis=1).to_string())
    print()
    print(real.groupby("source").agg(olay=("sure", "count"), sure_p50=("sure", "median"), genlik_p50=("genlik", "median"),
                                     baslangic=("baslangic", "median"), sakinlesme=("varyans_oran", lambda v: round((v < 0.5).mean(), 2)),
                                     kalici=("kalici", "mean")).round(2).to_string())
    print(f"\nbanka: {len(bank)} şablon → {HAVUZ / 'anomali_bankasi.npz'}")
