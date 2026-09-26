"""
Etiket denetimi: havuzdaki etiketli serilerde modelin bağlamla ayırt edemeyeceği etiketleri bulur.

    python etiket_denetimi.py              # tüm etiketli kaynaklar → data/havuz/etiket_denetimi.csv + özet
    python etiket_denetimi.py w3 lbnl      # yalnız bu kaynaklar

Neden: 3W (%88 pozitif, kararlı arıza), TEP (%96) ve LBNL'de olay pencereden (4096) uzun ya da önünde normal bağlam
yoktu; model "sapma yokken 1" öğreniyor, doğrulama AUC-PR'ı anomali oranına eşitleniyordu. Yeni kaynak havuza
girmeden önce bu betik çalıştırılır; bayraklı kaynaklar için olay_basi() ile etiket olay başlangıcına indirgenir.

Bayraklar (satır etiketi: herhangi bir hücre 1 → 1, tümü 0 → 0, aksi -1):
    oran        bilinen satırların %50'sinden fazlası pozitif (normal bağlam azınlıkta)
    uzun_olay   en uzun pozitif blok > PENCERE (tamamı pozitif pencereler oluşur)
    bagsiz      ilk olaydan önce min(MIN_BAGLAM, n/10)'dan az bilinen normal satır var
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
HAVUZ = ROOT / "data" / "havuz"
PENCERE = 4096
MIN_BAGLAM = 256


def satir_etiketi(L):
    return np.where((L == 1).any(1), 1, np.where((L == 0).all(1), 0, -1)).astype(np.int8)


def olcu(r):
    known = r >= 0
    pos = np.flatnonzero(r == 1)
    if not len(pos):
        return dict(oran=0.0, olay=0, en_uzun=0, bag=int(known.sum()))
    starts = pos[np.r_[True, np.diff(pos) > 1]]
    ends = pos[np.r_[np.diff(pos) > 1, True]] + 1
    return dict(oran=float((r == 1).sum() / max(known.sum(), 1)), olay=len(starts), en_uzun=int((ends - starts).max()),
                bag=int((r[:starts[0]] == 0).sum()))


def denetle(kaynaklar=None):
    kat = pd.read_csv(HAVUZ / "katalog.csv")
    kat = kat[(~kat.unlabeled) & (kat.benchmark.fillna("") == "")]
    if kaynaklar:
        kat = kat[kat.source.isin(kaynaklar)]
    rows = []
    for file, g in kat.groupby("file"):
        # Dosya parça parça okunur; seriler dosyada bitişik (_yaz sırayla yazar). "in" filtresi binlerce kimlikte çok yavaş.
        meta = g.set_index("series_id")
        want = set(meta.index)
        cur, buf = None, []

        def bitir(sid, parts):
            if sid in want:
                T, k = int(meta.at[sid, "n_rows"]), int(meta.at[sid, "n_channels"])
                m = olcu(satir_etiketi(np.concatenate(parts).reshape(T, k)))
                rows.append(dict(series_id=sid, source=meta.at[sid, "source"], n=T, **m))

        for batch in pq.ParquetFile(HAVUZ / "gercek" / file).iter_batches(batch_size=5_000_000, columns=["series_id", "label"]):
            sid = np.asarray(batch.column(0).to_pylist() if not hasattr(batch.column(0), "dictionary")
                             else batch.column(0).dictionary.to_numpy(zero_copy_only=False)[batch.column(0).indices.to_numpy()])
            lab = batch.column(1).to_numpy()
            cuts = np.flatnonzero(sid[1:] != sid[:-1]) + 1
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(sid)]):
                s_ = sid[a]
                if s_ != cur:
                    if cur is not None:
                        bitir(cur, buf)
                    cur, buf = s_, []
                if s_ in want:
                    buf.append(lab[a:b])
        if cur is not None:
            bitir(cur, buf)
        print(f"  {file}: {sum(r['series_id'] in want for r in rows)} seri", flush=True)
    df = pd.DataFrame(rows)
    df["b_oran"] = df.oran > 0.5
    df["b_uzun_olay"] = df.en_uzun > PENCERE
    df["b_bagsiz"] = (df.olay > 0) & (df.bag < np.minimum(MIN_BAGLAM, df.n // 10))   # kısa seride göreli (C-MAPSS, msft)
    df["bayrak"] = df.b_oran | df.b_uzun_olay | df.b_bagsiz
    return df


def olay_basi(lab, onset=2048):
    """Her kesintisiz pozitif bloğun ilk `onset` satırı 1, kalanı -1 (bağlamla ayırt edilebilen kısım)."""
    lab = lab.copy()
    idx = np.flatnonzero(lab == 1)
    if len(idx):
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        run_start = starts[np.searchsorted(starts, idx, side="right") - 1]
        lab[idx[idx - run_start >= onset]] = -1
    return lab


def baglam_duzelt(lab, onset=2048, min_baglam=MIN_BAGLAM):
    """olay_basi + önünde min(min_baglam, n/10) bilinen normal satır olmayan olaylar -1 (model sapmayı göremez).
    Satır etiketi (1-B) veya hücre etiketi (2-B, satır bazında karar) alır."""
    lab = np.asarray(lab).copy()
    row = lab if lab.ndim == 1 else satir_etiketi(lab)
    row = olay_basi(row, onset)
    need = min(min_baglam, len(row) // 10)
    idx = np.flatnonzero(row == 1)
    if len(idx):
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]] + 1
        normal = np.cumsum(row == 0)
        for a, b in zip(starts, ends):
            if (normal[a - 1] if a else 0) < need:
                row[a:b] = -1
    if lab.ndim == 1:
        return row
    lab[(row == -1)[:, None] & (lab == 1)] = -1
    return lab


if __name__ == "__main__":
    df = denetle(sys.argv[1:] or None)
    df.to_csv(HAVUZ / "etiket_denetimi.csv", index=False)
    ozet = df.groupby("source").agg(seri=("series_id", "size"), bayrakli=("bayrak", "sum"), oran_p50=("oran", "median"),
                                    en_uzun_max=("en_uzun", "max"), oran_bayrak=("b_oran", "sum"),
                                    uzun_olay=("b_uzun_olay", "sum"), bagsiz=("b_bagsiz", "sum"))
    pd.set_option("display.width", 200)
    print(ozet[ozet.bayrakli > 0].sort_values("bayrakli", ascending=False).to_string())
    print(f"\n{int(df.bayrak.sum())} / {len(df)} seri bayraklı → {HAVUZ / 'etiket_denetimi.csv'}")
