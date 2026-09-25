"""GPT-6 Astra: etiket güveni ve satır/hücre denetimini ayıran eğitim yardımcıları."""
import numpy as np

WEAK_SOURCES = {"cmapss", "ims_bearing", "femto"}


def label_confidence(meta):
    """Eski kataloglarda da zaman sınırından türetilen etiketleri tanır."""
    explicit = meta.get("label_confidence", None)
    if explicit is not None and np.isfinite(float(explicit)):
        return float(explicit)
    return 0.2 if meta.get("source") in WEAK_SOURCES else 0.3 if meta.get("source") in {"care", "care_c"} else 1.0


def row_targets(labels):
    """Herhangi bir bilinen pozitif → 1; tümü bilinen normal → 0; diğerleri bilinmiyor."""
    return np.where((labels == 1).any(1), 1,
                    np.where((labels == 0).all(1), 0, -1)).astype(np.int8)


def supervision(labels, level="cell", confidence=1.0, unknown_weight=0.05,
                rows=None, row_positive_visible=True):
    """Etiketsiz hücreler yalnızca düşük ağırlıklı arka plan varsayımıdır.

    Satır pozitifleri hücrelere yayılmaz. Satır negatifleri bütün hücrelerin normal
    olduğunu bildirir. Satır kaybı yalnızca pozitif satırlarda kullanılır; negatifler
    hücre kaybında bir kez sayılır.
    """
    labels = np.asarray(labels)
    target = np.where(labels >= 0, labels, 0).astype(np.int8)
    weight = np.where(labels >= 0, confidence, unknown_weight).astype(np.float32)
    r = row_targets(labels) if rows is None else np.asarray(rows, dtype=np.int8)
    rw = np.zeros(len(labels), dtype=np.float32)
    if level == "row":
        positive = r == 1
        target[positive] = -1
        weight[positive] = 0
        if row_positive_visible:
            rw[positive] = confidence
    return target, weight, r, rw


def normal_reference(X, labels, end, min_length=20, max_length=256):
    """Yalnızca pencere öncesindeki kesintisiz, doğrulanmış normal bölüm."""
    good = (labels[:end] == 0).all(1) & np.isfinite(X[:end]).all(1)
    boundaries = np.diff(np.r_[False, good, False].astype(np.int8))
    starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    eligible = np.flatnonzero(stops - starts >= min_length)
    if not len(eligible):
        return None
    i = eligible[-1]
    return X[max(starts[i], stops[i] - max_length):stops[i]].copy()
