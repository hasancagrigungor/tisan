"""
Zero-shot zaman serisi anomali modeli için çok alanlı eğitim verisi üreticisi.

14 alan: finans, imalat, uzay (uydu telemetrisi), hasta takibi, EKG, sürekli
glikoz ölçümü, biyoloji (biyoreaktör), enerji, bilişim (sunucu izleme), su
şebekesi, çevre/iklim, perakende, otomotiv, telekom, tarım, kimyasal proses
ve alandan bağımsız soyut seriler.

Her alanın kendine özgü:
  - fiziksel ölçekleri, birimleri ve sınırları (SpO2 en fazla 100, CPU 0-100 ...)
  - normal ama "anomali gibi görünen" davranışları (vardiya duruşları, uydu
    tutulmaları, yemek sonrası glikoz yükselişi, GC testere dişi, sulama ...)
  - gerçekçi arıza senaryoları (rulman aşınması, flash crash, bellek sızıntısı,
    ektopik kalp atışı, su kaçağı, kontaminasyon ...)
vardır. Alana özgü senaryoların yanında her alana genel anomaliler de eklenir.

Model hiçbir zaman alan adını görmez; alan bilgisi sadece çeşitlilik içindir.

Kullanım:
    rng = np.random.default_rng(0)
    ornek = make_sample(rng, difficulty=0.5)
    ornek["raw"]      # (T, 1+k) kullanıcı formatı: ilk sütun unix zaman
    ornek["values"]   # (2048, 100) model girdisi
    ornek["labels"]   # (2048, 100) hücre etiketleri

Kendi gerçek verinizi eklemek için: gen_* imzasında bir fonksiyon yazıp
DOMAINS sözlüğüne ekleyin (bkz. gen_real_template).
"""
import numpy as np

try:
    from scipy.signal import lfilter as _lfilter
except ImportError:  # scipy yoksa yavaş ama çalışan yedek yol
    _lfilter = None

MAX_CH = 100               # sütunlar her zaman 100'e tamamlanır
MAX_T = 4096              # modelin gördüğü pencere uzunluğu
MIN_T = 20                 # bundan kısa veri üretilmez
CLEAN_RATIO = 0.25         # örneklerin bu kadarında hiç anomali yok
DOMAIN_SCENARIO_RATIO = 0.7  # anomalilerin bu kadarı alana özgü, kalanı genel
MISSING_DATA_RATIO = 0.12  # anomalili örneklerde veri boşluğu olasılığı
PERSISTENT_RATIO = 0.10    # genel anomalilerde "sona kadar sürme" olasılığı (gerçek olaylarda ~%0; kalıcı senaryo seyrek)
PERSISTENCE_SCALE = 0.4    # alan senaryolarındaki kalıcılık olasılıklarının genel çarpanı

TYPES = ["normal", "spike", "level_shift", "flatline", "drift", "noise_burst",
         "pattern_change", "correlation_break", "missing_data"]
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
GENERIC_KINDS = ["spike", "level_shift", "flatline", "drift", "noise_burst",
                 "pattern_change", "correlation_break"]
PERSISTENT_OK = {"level_shift", "flatline", "drift", "noise_burst", "pattern_change"}


# =============================================================================
# Yardımcılar
# =============================================================================
def _step(t):
    return float(np.median(np.diff(t))) if len(t) > 1 else 1.0


def _hours(t):
    return (t % 86400) / 3600.0


def _weekday(t):  # 0 = pazartesi
    return ((t // 86400).astype(np.int64) + 3) % 7


def _doy(t):
    return (t / 86400.0) % 365.25


def _bump(h, center, width):
    """24 saatlik dairesel Gauss tepesi (günlük profiller için)."""
    d = (h - center + 12) % 24 - 12
    return np.exp(-0.5 * (d / width) ** 2)


def _lowpass(u, a):
    """Birinci derece filtre: y[i] = a*y[i-1] + (1-a)*u[i], y[0] = u[0]."""
    u = np.asarray(u, float)
    if _lfilter is not None:
        y, _ = _lfilter([1 - a], [1, -a], u, axis=0, zi=a * u[:1])
        return y
    y = np.empty_like(u)
    y[0] = u[0]
    for i in range(1, len(u)):
        y[i] = a * y[i - 1] + (1 - a) * u[i]
    return y


def _ar1(rng, T, tau_steps, sigma):
    """Durağan AR(1) gürültü. tau_steps: korelasyon süresi (adım), sigma: std."""
    phi = float(np.exp(-1.0 / max(tau_steps, 1e-6)))
    e = rng.normal(0, 1, T) * sigma * np.sqrt(max(1 - phi ** 2, 1e-12))
    e[0] = rng.normal(0, sigma)
    if _lfilter is not None:
        return _lfilter([1.0], [1.0, -phi], e)
    y = e.copy()
    for i in range(1, T):
        y[i] = phi * y[i - 1] + e[i]
    return y


def robust_std(x):
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    return mad if mad > 1e-8 else x.std() + 1e-8


class _Cols:
    """Üreticilerin sütun biriktirme yardımcısı."""
    def __init__(self):
        self.data, self.roles, self.ent = [], [], []

    def add(self, x, role, e):
        self.data.append(np.asarray(x, float))
        self.roles.append(role)
        self.ent.append(e)
        return len(self.data) - 1

    def __len__(self):
        return len(self.data)

    def out(self, k):
        return np.column_stack(self.data[:k]), list(self.roles[:k]), np.array(self.ent[:k])


# =============================================================================
# Anomali bağlamı: senaryoların ortak araçları
# =============================================================================
class Ctx:
    def __init__(self, rng, X, t, roles, ent, difficulty, extra, keep):
        self.rng, self.X, self.t = rng, X, t
        self.roles, self.ent, self.extra, self.keep = roles, ent, extra, keep
        self.T, self.k = X.shape
        self.dt = _step(t)
        self.f = 1.5 - difficulty            # 1.5 = belirgin, 0.5 = ince anomali
        self.labels = np.zeros((self.T, self.k), dtype=np.int8)
        self.types = np.zeros((self.T, self.k), dtype=np.int8)

    # --- sütun seçimi ---
    def cols(self, role=None, entity=None):
        return [c for c in range(self.k)
                if (role is None or self.roles[c] == role)
                and (entity is None or self.ent[c] == entity)]

    def col(self, role, entity):
        c = self.cols(role, entity)
        return c[0] if c else None

    def pick(self, cols):
        return int(self.rng.choice(cols)) if cols else None

    def pick_entity(self, *roles):
        ents = [e for e in np.unique(self.ent)
                if all(self.col(r, e) is not None for r in roles)]
        return int(self.rng.choice(ents)) if ents else None

    # --- zaman seçimi ---
    def segment(self, min_len=5, max_frac=0.3, persistent_prob=0.0):
        T, rng = self.T, self.rng
        if persistent_prob and rng.random() < persistent_prob * PERSISTENCE_SCALE:
            lo = max(T // 10, int(T * 0.3))    # önce en az %30 normal bağlam
            s = int(rng.integers(lo, max(T - 4, lo + 1)))
            return s, T
        max_len = max(min_len, int(T * max_frac))
        L = int(rng.integers(min_len, max_len + 1))
        L = max(1, min(L, T - T // 10 - 1))
        s = int(rng.integers(T // 10, T - L + 1))
        return s, s + L

    def rise(self, L, frac=0.2):
        """Hızlı yükselip sabit kalan şekil (0 -> 1)."""
        return 1 - np.exp(-np.arange(L) / max(L * frac, 1.0))

    def bell(self, L):
        """Yükselip geri dönen şekil (0 -> 1 -> 0)."""
        return np.sin(np.linspace(0, np.pi, L))

    def sd(self, c):
        return robust_std(self.X[:, c])

    def align(self, arr):
        return arr if self.keep is None else arr[self.keep]

    # --- etiketleme ---
    def mark(self, s, e, cols, kind):
        self.labels[s:e, cols] = 1
        self.types[s:e, cols] = TYPE_ID[kind]

    def mark_rows(self, rows, cols, kind):
        for c in cols:
            self.labels[rows, c] = 1
            self.types[rows, c] = TYPE_ID[kind]


# =============================================================================
# Genel (alandan bağımsız) anomaliler
# =============================================================================
# -----------------------------------------------------------------------------
# Gerçek anomali bankası: eğitim rolündeki etiketli gerçek olaylardan normalize şablonlar
# (anomali_istatistik.py üretir). Şablon = [önce 2L | olay L | sonra ≤L] × etkilenen sütunlar, MAD birimi.
# Hedef seriye ölçeklenip eklenir; hangi seride olduğu değil, bağlamdan sapması öğrenilir.
# -----------------------------------------------------------------------------
import os as _os
BANK_PATH = _os.environ.get("ANOMALI_BANKASI", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "havuz", "anomali_bankasi.npz"))
BANK_RATIO = 0.3           # genel enjeksiyonların bu kadarı bankadan (banka yoksa 0)
_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        if _os.path.exists(BANK_PATH):
            d = np.load(BANK_PATH, allow_pickle=True)
            _BANK = [dict(z=z, pre=int(p), L=int(L)) for z, p, L in zip(d["z"], d["pre"], d["L"])]
        else:
            _BANK = []
    return _BANK


def bank_anomaly(ctx, min_effect=0.3):
    """Bankadan bir gerçek olay şablonu alır, zaman ölçeğini rastgele eğer (×0.5–2), hedef sütun(lar)ın
    MAD'ına ölçekler ve olay + sonrası bölümünü ekler. Etiket: olay satırları, etkilenen hedef sütunlar."""
    bank = _bank()
    if not bank:
        return False
    rng, X, T, k = ctx.rng, ctx.X, ctx.T, ctx.k
    tpl = bank[int(rng.integers(len(bank)))]
    z = tpl["z"][tpl["pre"]:]                                   # olay + sonrası (önce bölümü ≈ 0 sapma)
    L0 = tpl["L"]
    scale_t = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    n_new = max(2, int(round(len(z) * scale_t)))
    L = max(1, int(round(L0 * scale_t)))
    if n_new >= T - T // 10 - 1:
        return False
    idx = np.linspace(0, len(z) - 1, n_new)
    zr = np.column_stack([np.interp(idx, np.arange(len(z)), z[:, j]) for j in range(z.shape[1])])
    s = int(rng.integers(T // 10, T - n_new))
    e = s + L
    m = min(k, zr.shape[1])
    cols = rng.choice(k, m, replace=False)
    tcols = rng.choice(zr.shape[1], m, replace=False)
    if ctx.labels[s:e, cols].any():
        return False
    gain = float(np.exp(rng.normal(0, 0.3))) * ctx.f
    labeled = []
    for c, j in zip(cols, tcols):
        sd = ctx.sd(c)
        before = X[s:s + n_new, c].copy()
        X[s:s + n_new, c] += zr[:, j] * sd * gain
        eff = np.mean(np.abs(X[s:e, c] - before[:L])) / max(sd, 1e-9)
        if eff < min_effect:
            X[s:s + n_new, c] = before
        else:
            labeled.append(int(c))
    if not labeled:
        return False
    ctx.mark(s, e, labeled, "pattern_change")
    ctx.types[s:e, labeled] = -1                                        # gerçek olayın türü bilinmiyor: tür kaybına girmez
    return True


def _partner(X, c, min_corr=0.5):
    """c ile en güçlü doğrusal ilişkili sütun; |korelasyon| < min_corr ise None."""
    if X.shape[1] < 2:
        return None
    x = X[:, c]
    if x.std() < 1e-9:
        return None
    best, best_r = None, 0.0
    for j in range(X.shape[1]):
        if j == c or X[:, j].std() < 1e-9:
            continue
        r = abs(np.corrcoef(x, X[:, j])[0, 1])
        if np.isfinite(r) and r > best_r:
            best, best_r = j, r
    return best if best_r >= min_corr else None


def generic_anomaly(ctx, kind=None, min_effect=0.3):
    """Alandan bağımsız anomali enjeksiyonu. Uygunluk (önce) ve etki (sonra) kontrolü yapar;
    geçersizse veriyi geri alır, etiket vermez ve False döner. Etiketli bölgeyle üst üste binmez."""
    rng, X, T, k = ctx.rng, ctx.X, ctx.T, ctx.k
    if kind is None and _bank() and rng.random() < BANK_RATIO:
        return bank_anomaly(ctx, min_effect)
    kind = kind or str(rng.choice(GENERIC_KINDS))
    if kind == "correlation_break":
        cands = [c for c in range(k) if _partner(X, c) is not None]
        if not cands:
            kind = "spike"
    c = int(rng.choice(cands)) if kind == "correlation_break" else int(rng.integers(k))
    x = X[:, c]
    sd = ctx.sd(c)
    strength = float(np.exp(rng.normal(np.log(4.0), 0.8))) * ctx.f * 1.6   # log-normal: p50 ≈ 4·f, ağır kuyruk (gerçek: 2.4 / 44 MAD)
    sign = rng.choice([-1, 1])
    if kind == "spike":
        s = int(rng.integers(T // 10, T - 1))
        e = min(T, s + int(rng.integers(1, 4)))
    else:
        s, e = ctx.segment(5, 0.3, PERSISTENT_RATIO if kind in PERSISTENT_OK else 0.0)
    L = e - s
    seg = slice(s, e)
    if ctx.labels[s:e, c].any():                       # üst üste binme: önceki etiket bozulmasın
        return False
    variant = rng.random()      # gerçek arızalara benzeyen ince alt varyantlar (~%35)
    # 3) çok sütunlu olay: gerçek anomaliler sütunların ~yarısını etkiliyor → %40 olasılıkla ilişkili bir sütun grubuna aynı bozulma
    if k > 1 and kind != "correlation_break" and rng.random() < 0.4:
        n_extra = int(rng.integers(1, max(2, k // 2) + 1))
        corr = np.array([abs(np.corrcoef(X[:, c], X[:, j])[0, 1]) if j != c and X[:, j].std() > 1e-9 else -1 for j in range(k)])
        corr[~np.isfinite(corr)] = -1
        group = [c] + [int(j) for j in np.argsort(-corr)[:n_extra] if corr[j] > 0 and not ctx.labels[s:e, j].any()]
        if len(group) > 1:
            ok_any = False
            for j in group[1:]:
                ok_any |= _inject_column(ctx, kind, j, s, e, sign, strength, rng.random(), min_effect)
            # birincil sütun aşağıda işlenir
    return _inject_column(ctx, kind, c, s, e, sign, strength, variant, min_effect)


def _inject_column(ctx, kind, c, s, e, sign, strength, variant, min_effect=0.3):
    """Tek sütuna enjeksiyon gövdesi; etki kontrolüyle etiketler, geçersizse geri alır."""
    rng, X, T, k = ctx.rng, ctx.X, ctx.T, ctx.k
    x = X[:, c]
    sd = ctx.sd(c)
    L = e - s
    seg = slice(s, e)
    if kind in ("flatline", "pattern_change") and robust_std(x[seg]) < 1e-6:   # zaten sabit bölge
        return False
    before = x[seg].copy()
    if kind == "spike":
        x[seg] += sign * strength * sd
    elif kind == "level_shift":
        if variant < 0.35:      # aralık içi set noktası kayması: küçük ama kalıcı (HAI tipi)
            x[seg] += sign * rng.uniform(0.8, 1.8) * sd * ctx.f
        else:
            x[seg] += sign * strength * sd * 0.6
    elif kind == "flatline":
        if variant < 0.35:      # sensör takılması, sonra gerçek değere sıçrayarak dönüş
            x[seg] = x[s] + rng.normal(0, 0.02 * sd, L)
        elif variant < 0.65:    # DÖNGÜ DURMASI: çevrimsel süreç uç bir seviyede takılı kalır (kompresör sürekli yükte, pompa açık kalır)
            lvl = np.quantile(x, rng.choice([0.05, 0.1, 0.9, 0.95]))
            x[seg] = lvl + rng.normal(0, 0.03 * sd, L)
        else:
            x[seg] = x[s]
    elif kind == "drift":
        if variant < 0.35:      # kademeli bozulma: yavaş kayma + artan salınım (rulman, pompa)
            ramp = np.linspace(0, 1, L)
            x[seg] += sign * strength * sd * 0.5 * ramp ** 2 + rng.normal(0, 1, L) * strength * sd * 0.4 * ramp
        else:
            x[seg] += sign * strength * sd * np.linspace(0, 1, L)
    elif kind == "noise_burst":
        if variant < 0.25:      # varyans artışı yavaş yavaş gelir (aşınma)
            x[seg] += rng.normal(0, 1, L) * strength * sd * 0.5 * np.linspace(0.2, 1, L)
        elif variant < 0.5:     # varyans ÇÖKÜŞÜ: seri sakinleşir (sunucu arızası, akış durması); "sakin = normal" kısayolunu kırar
            base = np.median(x[max(0, s - L):s]) if s > 0 else np.median(x[seg])
            x[seg] = base + (x[seg] - np.median(x[seg])) * rng.uniform(0.0, 0.15)
        else:
            x[seg] += rng.normal(0, strength * sd * 0.5, L)
    elif kind == "pattern_change":
        p = rng.uniform(3, max(4.0, L / 2))
        x[seg] = np.median(x[seg]) + 1.5 * sd * np.sin(2 * np.pi * np.arange(L) / p)
    elif kind == "correlation_break":
        # gerçekten ilişkili olduğu sütunla bağı kopar: kendi geçmişinden kopya ya da ters işaretli izleme
        j = _partner(X, c)
        if variant < 0.5:
            s2 = int(rng.integers(0, T - L + 1))
            if abs(s2 - s) < L:
                s2 = (s + L + int(rng.integers(0, T))) % (T - L + 1)
            x[seg] = x[s2:s2 + L].copy()
        else:
            partner = X[seg, j]
            x[seg] = np.median(x[seg]) - (partner - np.median(partner)) * (sd / max(robust_std(partner), 1e-9))
    # etki kontrolü: değişim serinin oynaklığına göre anlamlı değilse geri al
    effect = np.mean(np.abs(x[seg] - before)) / max(sd, 1e-9)
    if kind == "flatline":
        effect = robust_std(before) / max(sd, 1e-9)         # sabitleme: önceki oynaklık kaybı
    if not np.isfinite(effect) or effect < min_effect:
        x[seg] = before
        return False
    ctx.mark(s, e, [c], kind)
    return True


def safe_inject(ctx, op, blocked=None, **kw):
    """Enjeksiyon sarmalayıcısı: op(ctx) çalıştırılır; yeni etiket üretmediyse, etiketlediği hücreler
    `blocked` (dokunulmaması gereken gerçek pozitif bölge) ile çakışıyorsa ya da etiketlediği hücrelerin
    çoğu değişmediyse veri/etiket geri alınır ve False döner."""
    X0, L0, T0 = ctx.X.copy(), ctx.labels.copy(), ctx.types.copy()
    try:
        ok = op(ctx, **kw)
    except Exception:
        ok = False
    new = (ctx.labels == 1) & (L0 != 1)
    changed = np.abs(ctx.X - X0) > 1e-9
    bad = (not ok) or (not new.any()) \
        or (blocked is not None and (new & np.asarray(blocked, dtype=bool)).any()) \
        or ((new & ~changed).sum() > 0.5 * new.sum() and not (ctx.types[new] == TYPE_ID["flatline"]).all())
    if bad:
        ctx.X[:], ctx.labels[:], ctx.types[:] = X0, L0, T0
        return False
    return True


# =============================================================================
# 1. FİNANS: hisse fiyatları ve işlem hacimleri
# =============================================================================
def gen_finance(rng, t, k, extra):
    T, dt = len(t), _step(t)
    yrs = 1 / 252 if dt >= 86400 else dt / (365 * 86400)
    log_vol = _ar1(rng, T, 30, 0.35)                      # oynaklık kümelenmesi
    market = rng.standard_t(4, T) / np.sqrt(2) * rng.uniform(0.1, 0.35) * np.sqrt(yrs) * np.exp(log_vol)
    cols, e = _Cols(), 0
    while len(cols) < k:
        sig = rng.uniform(0.1, 0.6) * np.sqrt(yrs)
        r = rng.uniform(0.3, 1.6) * market + rng.standard_t(rng.uniform(3, 8), T) * 0.7 * sig * np.exp(0.5 * log_vol)
        r[0] = 0.0
        price = np.maximum(np.round(10 ** rng.uniform(0, 3.5) * np.exp(np.cumsum(r)), 2), 0.01)
        z = np.abs(r) / (r.std() + 1e-12)
        per_step = 10 ** rng.uniform(4, 7) * min(1.0, dt / 86400)
        vol = np.round(per_step * np.exp(0.5 * z + _ar1(rng, T, 10, 0.3)))
        cols.add(price, "price", e)
        cols.add(vol, "volume", e)
        e += 1
    return cols.out(k)


def sc_flash_crash(ctx):
    """Piyasa genelinde ani çöküş ve kısmi toparlanma; hacimler patlar."""
    P = ctx.cols("price")
    if not P:
        return False
    rng = ctx.rng
    s, e = ctx.segment(3, 0.08)
    L = e - s
    k1 = max(1, L // 3)
    rest = rng.uniform(0, 0.3)
    curve = np.concatenate([np.linspace(0, 1, k1, endpoint=False),
                            np.linspace(1, rest, L - k1), np.full(ctx.T - e, rest)])
    depth = rng.uniform(0.04, 0.15) * ctx.f
    hit = [c for c in P if rng.random() < 0.8] or [P[0]]
    for c in hit:
        ctx.X[s:, c] *= 1 - depth * rng.uniform(0.6, 1.4) * curve
        ctx.mark(s, e, [c], "spike")
        v = ctx.col("volume", ctx.ent[c])
        if v is not None:
            ctx.X[s:e, v] *= rng.uniform(3, 10)
            ctx.mark(s, e, [v], "spike")
    return True


def sc_trading_halt(ctx):
    """İşlem durdurma: fiyat donar, hacim sıfırlanır."""
    c = ctx.pick(ctx.cols("price"))
    if c is None:
        return False
    s, e = ctx.segment(5, 0.3, persistent_prob=0.3)
    ctx.X[s:e, c] = ctx.X[s, c]
    ctx.mark(s, e, [c], "flatline")
    v = ctx.col("volume", ctx.ent[c])
    if v is not None:
        ctx.X[s:e, v] = 0
        ctx.mark(s, e, [v], "flatline")
    return True


def sc_fat_finger(ctx):
    """Hatalı emir veya bozuk veri: tek bir absürt fiyat."""
    c = ctx.pick(ctx.cols("price"))
    if c is None:
        return False
    i = int(ctx.rng.integers(ctx.T // 10, ctx.T))
    ctx.X[i, c] *= 1 + ctx.rng.choice([-1, 1]) * ctx.rng.uniform(0.05, 0.3) * ctx.f
    ctx.mark(i, i + 1, [c], "spike")
    return True


def sc_volatility_burst(ctx):
    """Tek bir varlıkta ani oynaklık artışı."""
    c = ctx.pick(ctx.cols("price"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.3)
    lp = np.log(np.maximum(ctx.X[:, c], 1e-6))
    r = np.diff(lp)
    r[s - 1:e - 1] *= ctx.rng.uniform(3, 8) * ctx.f
    ctx.X[:, c] = np.exp(lp[0] + np.concatenate([[0.0], np.cumsum(r)]))
    ctx.mark(s, e, [c], "noise_burst")
    return True


# =============================================================================
# 2. İMALAT: dönen makineler (motor, pompa, fan)
# =============================================================================
def gen_manufacturing(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h, wd = _hours(t), _weekday(t)
    amb = 22 + 3 * np.sin(2 * np.pi * (h - 9) / 24) + _ar1(rng, T, 3 * 3600 / dt, 0.8)
    cols, e = _Cols(), 0
    while len(cols) < k:
        mode = rng.integers(3)                    # 7/24, iki vardiya, tek vardiya
        if mode == 0:
            on = np.ones(T)
        elif mode == 1:
            on = ((h >= 6) & (h < 22) & (wd < 6)).astype(float)
        else:
            on = ((h >= 8) & (h < 17) & (wd < 5)).astype(float)
        on = _lowpass(on, np.exp(-dt / 60))       # planlı duruşlar NORMALDİR
        load = np.clip(0.65 + _ar1(rng, T, 1800 / dt, 0.12), 0.1, 1.0)
        rpm = np.round(on * rng.choice([750, 1000, 1500, 3000]) * (1 + 0.004 * rng.normal(0, 1, T)))
        i_nom = rng.uniform(5, 200)
        cur = np.round(np.maximum(on * i_nom * (0.35 + 0.65 * load) + rng.normal(0, 0.01 * i_nom, T), 0), 1)
        vib = np.round(0.05 + on * rng.uniform(0.5, 4) * (0.6 + 0.4 * load) * np.exp(rng.normal(0, 0.08, T)), 3)
        heat = _lowpass(on * (0.5 + 0.5 * load), np.exp(-dt / 1800))
        temp = np.round(amb + rng.uniform(15, 45) * heat + rng.normal(0, 0.2, T), 1)
        for x, role in [(rpm, "rpm"), (cur, "current"), (vib, "vibration"), (temp, "temperature")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_bearing_wear(ctx):
    """Rulman aşınması: titreşim giderek artar, sıcaklık yavaşça yükselir."""
    e_ = ctx.pick_entity("vibration")
    if e_ is None:
        return False
    rng, X = ctx.rng, ctx.X
    cv, ct = ctx.col("vibration", e_), ctx.col("temperature", e_)
    s, e = ctx.segment(20, 0.5, persistent_prob=0.6)
    L = e - s
    ramp = np.linspace(0, 1, L) ** 2
    X[s:e, cv] = X[s:e, cv] * (1 + rng.uniform(1, 4) * ctx.f * ramp) \
        + np.abs(rng.normal(0, 1, L)) * ramp * ctx.sd(cv) * ctx.f
    ctx.mark(s, e, [cv], "drift")
    if ct is not None:
        X[s:e, ct] += rng.uniform(4, 15) * ctx.f * ramp
        ctx.mark(s, e, [ct], "drift")
    return True


def sc_unexpected_stop(ctx):
    """Çalışma saatinde plansız duruş."""
    e_ = ctx.pick_entity("rpm")
    if e_ is None:
        return False
    cr = ctx.col("rpm", e_)
    on_idx = np.where(ctx.X[:, cr] > 0)[0]
    on_idx = on_idx[(on_idx >= ctx.T // 10) & (on_idx < ctx.T - 5)]
    if len(on_idx) < 5:
        return False
    s = int(ctx.rng.choice(on_idx))
    off_after = np.where(ctx.X[s:, cr] <= 0)[0]
    run_end = s + off_after[0] if len(off_after) else ctx.T
    e = ctx.T if ctx.rng.random() < 0.3 else s + int(ctx.rng.integers(5, max(6, ctx.T // 4)))
    e = min(e, run_end)
    for role, val in [("rpm", 0.0), ("current", 0.0), ("vibration", 0.05)]:
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] = val
            ctx.mark(s, e, [c], "level_shift")
    return True


def sc_current_mismatch(ctx):
    """Devir normal ama akım farklı: mekanik sürtünme veya elektrik arızası."""
    c = ctx.pick(ctx.cols("current"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.3, persistent_prob=0.3)
    ctx.X[s:e, c] *= 1 + ctx.rng.choice([-1, 1]) * ctx.rng.uniform(0.2, 0.5) * ctx.f
    ctx.mark(s, e, [c], "correlation_break")
    return True


def sc_overheat(ctx):
    """Soğutma arızası: sıcaklık yeni ve yüksek bir seviyeye çıkar."""
    c = ctx.pick(ctx.cols("temperature"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.5)
    ctx.X[s:e, c] += ctx.rng.uniform(6, 20) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 3. UZAY: uydu telemetrisi (yörünge döngüleri, tutulmalar)
# =============================================================================
def gen_space(rng, t, k, extra):
    T, dt = len(t), _step(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        P = rng.uniform(88, 110) * 60                     # yörünge periyodu (s)
        ecl = rng.uniform(0, 0.4)                         # tutulma oranı (0: tutulma sezonu dışı)
        phase = (t / P + rng.uniform()) % 1
        sun = _lowpass((phase >= ecl).astype(float), np.exp(-dt / 60))
        imax = rng.uniform(5, 40)
        sol = np.round(np.maximum(sun * imax * (0.95 + 0.05 * np.cos(2 * np.pi * phase))
                                  + rng.normal(0, 0.01 * imax, T), 0), 2)
        batt = np.round(26 + 2.5 * _lowpass(sun, np.exp(-dt / (0.25 * P))) + rng.normal(0, 0.02, T), 3)
        temp = np.round(-60 + 110 * _lowpass(sun, np.exp(-dt / (0.08 * P))) + rng.normal(0, 0.4, T), 1)
        # Reaksiyon tekeri: momentum birikir, periyodik boşaltmalar NORMALDİR
        rate, lim, w0 = rng.uniform(-0.5, 0.5), rng.uniform(1500, 3500), rng.uniform(-500, 500)
        w, cur = np.empty(T), w0 + rng.uniform(-0.5, 0.5) * lim
        for i in range(T):
            cur += rate * dt
            if abs(cur) > lim:
                cur = w0
            w[i] = cur
        wheel = np.round(w + 40 * np.sin(2 * np.pi * phase) + rng.normal(0, 2, T), 1)
        att = np.round(np.abs(rng.normal(0, rng.uniform(1, 5), T)) + 0.5 * (1 + np.sin(2 * np.pi * phase)), 2)
        for x, role in [(sol, "solar_current"), (batt, "battery_voltage"), (temp, "panel_temp"),
                        (wheel, "wheel_speed"), (att, "attitude_error")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_seu(ctx):
    """Kozmik ışın kaynaklı bit hatası (single-event upset): tek, aşırı değer."""
    c = int(ctx.rng.integers(ctx.k))
    i = int(ctx.rng.integers(ctx.T // 10, ctx.T))
    ctx.X[i, c] += ctx.rng.choice([-1, 1]) * ctx.sd(c) * ctx.rng.uniform(15, 60)
    ctx.mark(i, i + 1, [c], "spike")
    return True


def sc_battery_degradation(ctx):
    """Batarya yaşlanması: tutulma sırasındaki voltaj düşüşleri derinleşir."""
    e_ = ctx.pick_entity("battery_voltage", "solar_current")
    if e_ is None:
        return False
    cb, cs = ctx.col("battery_voltage", e_), ctx.col("solar_current", e_)
    s, e = ctx.segment(30, 0.6, persistent_prob=0.7)
    L = e - s
    ecl = (ctx.X[s:e, cs] < 0.1 * max(ctx.X[:, cs].max(), 1e-6)).astype(float)
    ramp = np.linspace(0.3, 1, L)
    if ecl.sum() > 0:
        ctx.X[s:e, cb] -= ctx.rng.uniform(0.3, 1.2) * ctx.f * ramp * ecl
        ctx.mark(s, e, [cb], "pattern_change")
    else:
        ctx.X[s:e, cb] -= ctx.rng.uniform(0.2, 0.6) * ctx.f * np.linspace(0, 1, L)
        ctx.mark(s, e, [cb], "drift")
    return True


def sc_wheel_friction(ctx):
    """Teker yatağında sürtünme: hız dalgalanır, yönelim hatası artar."""
    e_ = ctx.pick_entity("wheel_speed")
    if e_ is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.4)
    L = e - s
    cw, ca = ctx.col("wheel_speed", e_), ctx.col("attitude_error", e_)
    ctx.X[s:e, cw] += ctx.rng.normal(0, ctx.sd(cw) * ctx.rng.uniform(0.3, 1.0) * ctx.f, L)
    ctx.mark(s, e, [cw], "noise_burst")
    if ca is not None:
        ctx.X[s:e, ca] += np.abs(ctx.rng.normal(0, ctx.sd(ca) * ctx.rng.uniform(2, 5) * ctx.f, L))
        ctx.mark(s, e, [ca], "noise_burst")
    return True


def sc_heater_stuck(ctx):
    """Isıtıcı açık kaldı: panel sıcaklığı kalıcı olarak yükselir."""
    c = ctx.pick(ctx.cols("panel_temp"))
    if c is None:
        return False
    s, e = ctx.segment(15, 0.5, persistent_prob=0.5)
    ctx.X[s:e, c] += ctx.rng.uniform(10, 40) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_solar_string_failure(ctx):
    """Güneş paneli dizisi arızası: güneşteyken üretilen akım düşer."""
    c = ctx.pick(ctx.cols("solar_current"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.6)
    ctx.X[s:e, c] *= 1 - ctx.rng.uniform(0.15, 0.5) * ctx.f
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 4. HASTA TAKİBİ: yaşamsal bulgular
# =============================================================================
def gen_vitals(rng, t, k, extra):
    T, dt = len(t), _step(t)
    circ = np.cos(2 * np.pi * (_hours(t) - 16) / 24)      # sirkadiyen ritim
    cols, e = _Cols(), 0
    while len(cols) < k:
        hr = np.round(rng.uniform(60, 85) + 6 * circ + _ar1(rng, T, 600 / dt, 4))
        spo2 = np.clip(np.round(rng.uniform(96, 99) + _ar1(rng, T, 300 / dt, 0.6)), 88, 100)
        rr = np.round(rng.uniform(12, 18) + _ar1(rng, T, 600 / dt, 1.2))
        bt = np.round(rng.uniform(36.4, 36.9) + 0.3 * circ + _ar1(rng, T, 3600 / dt, 0.08), 1)
        bp = np.round(rng.uniform(105, 135) + 8 * circ + _ar1(rng, T, 900 / dt, 5))
        for x, role in [(hr, "heart_rate"), (spo2, "spo2"), (rr, "resp_rate"),
                        (bt, "body_temp"), (bp, "sys_bp")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_tachycardia(ctx):
    c = ctx.pick(ctx.cols("heart_rate"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.4)
    ctx.X[s:e, c] = np.round(ctx.X[s:e, c] + ctx.rng.uniform(20, 50) * ctx.f * ctx.rise(e - s))
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_desaturation(ctx):
    c = ctx.pick(ctx.cols("spo2"))
    if c is None:
        return False
    s, e = ctx.segment(8, 0.3)
    ctx.X[s:e, c] = np.clip(np.round(ctx.X[s:e, c] - ctx.rng.uniform(4, 12) * ctx.f * ctx.bell(e - s)), 50, 100)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_fever(ctx):
    c = ctx.pick(ctx.cols("body_temp"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.5)
    ctx.X[s:e, c] = np.round(ctx.X[s:e, c] + ctx.rng.uniform(1, 2.5) * ctx.f * np.linspace(0, 1, e - s), 1)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_probe_off(ctx):
    """Parmak probu çıktı: SpO2 ve nabız sıfıra düşer."""
    e_ = ctx.pick_entity("spo2")
    if e_ is None:
        return False
    s, e = ctx.segment(5, 0.3, persistent_prob=0.3)
    for role in ("spo2", "heart_rate"):
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] = 0
            ctx.mark(s, e, [c], "flatline")
    return True


def sc_motion_artifact(ctx):
    c = ctx.pick(ctx.cols("heart_rate"))
    if c is None:
        return False
    s, e = ctx.segment(5, 0.2)
    ctx.X[s:e, c] = np.round(np.maximum(ctx.X[s:e, c] + ctx.rng.normal(0, ctx.rng.uniform(10, 30) * ctx.f, e - s), 20))
    ctx.mark(s, e, [c], "noise_burst")
    return True


def sc_hypotension(ctx):
    """Tansiyon düşer, kalp telafi etmek için hızlanır: iki sütun birlikte."""
    e_ = ctx.pick_entity("sys_bp")
    if e_ is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.3)
    r = ctx.rise(e - s)
    cb, ch = ctx.col("sys_bp", e_), ctx.col("heart_rate", e_)
    ctx.X[s:e, cb] = np.round(ctx.X[s:e, cb] - ctx.rng.uniform(20, 40) * ctx.f * r)
    ctx.mark(s, e, [cb], "level_shift")
    if ch is not None:
        ctx.X[s:e, ch] = np.round(ctx.X[s:e, ch] + ctx.rng.uniform(10, 25) * ctx.f * r)
        ctx.mark(s, e, [ch], "level_shift")
    return True


# =============================================================================
# 5. EKG: yüksek frekanslı kalp dalga formu
# =============================================================================
_ECG_WAVES = [(-0.20, 0.15, 0.025), (-0.03, -0.15, 0.010), (0.0, 1.0, 0.012),
              (0.03, -0.25, 0.012), (0.25, 0.35, 0.050)]      # P, Q, R, S, T
_PVC_WAVES = [(0.0, 1.6, 0.035), (0.08, -0.6, 0.040), (0.30, -0.4, 0.060)]  # geniş QRS


def _ecg_wave(t, beats, pvc=()):
    y = np.zeros(len(t))
    for group, waves in ((beats, _ECG_WAVES), (pvc, _PVC_WAVES)):
        for b in group:
            for off, a, w in waves:
                y += a * np.exp(-0.5 * ((t - b - off) / w) ** 2)
    return y


def gen_ecg(rng, t, k, extra):
    T = len(t)
    info_all = extra.setdefault("ecg", {})
    cols, e = _Cols(), 0
    while len(cols) < k:
        rr = 60 / rng.uniform(55, 100)
        beats, b = [], t[0] - rng.uniform(0, rr)
        while b < t[-1] + 1:
            beats.append(b)
            b += rr * (1 + rng.normal(0, 0.03))       # kalp hızı değişkenliği
        beats = np.array(beats)
        wave = _ecg_wave(t, beats)
        scale = rng.uniform(0.5, 2.0)
        gains = [scale, scale * rng.uniform(0.4, 0.9), scale * rng.uniform(-0.6, 0.3)]
        info = {"beats": beats, "pvc": [], "gains": [], "other": [], "cols": []}
        for g in gains:
            if len(cols) >= k:
                break
            other = 0.05 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.3)) + rng.normal(0, 0.015, T)
            idx = cols.add(np.round(g * wave + other, 4), "ecg_lead", e)
            info["gains"].append(g)
            info["other"].append(other)
            info["cols"].append(idx)
        info_all[e] = info
        e += 1
    return cols.out(k)


def sc_ectopic_beat(ctx):
    """Erken ve geniş bir atım (PVC) ya da atlanan bir atım."""
    infos = ctx.extra.get("ecg", {})
    if not infos:
        return False
    e_ = int(ctx.rng.choice(list(infos)))
    info, t = infos[e_], ctx.t
    beats = info["beats"]
    valid = np.where((beats > t[ctx.T // 10]) & (beats < t[-1] - 0.3))[0]
    valid = valid[valid >= 1]
    if len(valid) == 0:
        return False
    j = int(ctx.rng.choice(valid))
    prev, rr = beats[j - 1], beats[j] - beats[j - 1]
    if ctx.rng.random() < 0.7:
        info["pvc"].append(prev + ctx.rng.uniform(0.55, 0.75) * rr)
    info["beats"] = np.delete(beats, j)
    for g, other, c in zip(info["gains"], info["other"], info["cols"]):
        ctx.X[:, c] = np.round(g * _ecg_wave(t, info["beats"], info["pvc"]) + ctx.align(other), 4)
    s = int(np.searchsorted(t, prev + 0.15 * rr))
    e = max(s + 1, int(np.searchsorted(t, beats[j] + 0.45 * rr)))
    ctx.mark(s, min(e, ctx.T), info["cols"], "pattern_change")
    return True


def sc_lead_off(ctx):
    c = ctx.pick(ctx.cols("ecg_lead"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.3)
    ctx.X[s:e, c] = ctx.rng.choice([0.0, ctx.X[:, c].max() * 3])   # sıfır ya da ray voltajı
    ctx.mark(s, e, [c], "flatline")
    return True


def sc_muscle_noise(ctx):
    c = ctx.pick(ctx.cols("ecg_lead"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4)
    ctx.X[s:e, c] += ctx.rng.normal(0, ctx.rng.uniform(0.1, 0.4) * ctx.f, e - s)
    ctx.mark(s, e, [c], "noise_burst")
    return True


# =============================================================================
# 6. SÜREKLİ GLİKOZ ÖLÇÜMÜ (CGM)
# =============================================================================
def gen_cgm(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h = _hours(t)
    d0, d1 = int(t[0] // 86400) - 1, int(t[-1] // 86400) + 1
    cols, e = _Cols(), 0
    while len(cols) < k:
        g = rng.uniform(90, 120) + 10 * _bump(h, 5, 1.5) + _ar1(rng, T, 3600 / dt, 8)
        for d in range(d0, d1 + 1):
            for mh in (7.5, 12.5, 19.0):          # yemek sonrası yükselişler NORMALDİR
                if rng.random() < 0.9:
                    tm = d * 86400 + (mh + rng.normal(0, 0.7)) * 3600
                    x = np.maximum((t - tm) / 3600, 0)
                    g = g + rng.uniform(30, 90) * (x / 0.75) * np.exp(1 - x / 0.75)
        cols.add(np.clip(np.round(g), 40, 400), "glucose", e)
        e += 1
    return cols.out(k)


def sc_hypoglycemia(ctx):
    c = ctx.pick(ctx.cols("glucose"))
    if c is None:
        return False
    s, e = ctx.segment(6, 0.3)
    w = ctx.bell(e - s) ** 0.5
    target = ctx.rng.uniform(45, 65)
    ctx.X[s:e, c] = np.round(ctx.X[s:e, c] * (1 - w) + target * w)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_compression_low(ctx):
    """Sensör üzerine yatılınca oluşan sahte, kısa düşüş."""
    c = ctx.pick(ctx.cols("glucose"))
    if c is None:
        return False
    s, e = ctx.segment(3, 0.05)
    ctx.X[s:e, c] = np.clip(np.round(ctx.X[s:e, c] - ctx.rng.uniform(40, 80) * ctx.f * ctx.bell(e - s)), 40, 400)
    ctx.mark(s, e, [c], "spike")
    return True


def sc_hyperglycemia(ctx):
    c = ctx.pick(ctx.cols("glucose"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.5, persistent_prob=0.3)
    ctx.X[s:e, c] = np.clip(np.round(ctx.X[s:e, c] + ctx.rng.uniform(80, 150) * ctx.f * ctx.rise(e - s)), 40, 400)
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 7. BİYOLOJİ: biyoreaktör / hücre kültürü
# =============================================================================
def gen_bioreactor(rng, t, k, extra):
    T, dt = len(t), _step(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        th = (t - t[0]) / 3600 + rng.uniform(-24, 72)       # aşılamadan beri geçen saat
        K, r, tmid = rng.uniform(5, 30), rng.uniform(0.08, 0.3), rng.uniform(20, 60)
        od = K / (1 + np.exp(-r * (th - tmid)))             # lojistik büyüme
        gn = (r * od * (1 - od / K)) / (r * K / 4)          # normalize büyüme hızı (0..1)
        od_m = np.round(np.maximum(od * (1 + rng.normal(0, 0.01, T)), 0), 3)
        temp = np.round(37 + 0.05 * np.sin(2 * np.pi * t / (rng.uniform(10, 30) * 60))
                        + rng.normal(0, 0.02, T), 2)
        ph, p = np.empty(T), 7.0 + rng.uniform(-0.03, 0.03)
        for i in range(T):                                   # asit birikir, baz eklenir: testere dişi
            p -= 0.004 * gn[i] * dt / 60 * rng.uniform(0.8, 1.2)
            if p < 6.95:
                p += 0.08
            ph[i] = p
        ph = np.round(ph + rng.normal(0, 0.005, T), 3)
        stir = 200 + 50 * np.round(16 * np.clip(gn * 1.1, 0, 1))   # kademeli karıştırıcı
        do = np.round(np.clip(35 + 60 * (1 - gn) + rng.normal(0, 1.5, T), 0, 100), 1)
        co2 = np.round(0.04 + 4 * gn + rng.normal(0, 0.03, T), 3)
        for x, role in [(od_m, "optical_density"), (temp, "culture_temp"), (ph, "ph"),
                        (do, "dissolved_oxygen"), (stir, "stirrer_rpm"), (co2, "co2_offgas")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_contamination(ctx):
    """Kontaminasyon: beklenmeyen büyüme, oksijen düşüşü, pH kayması."""
    e_ = ctx.pick_entity("optical_density")
    if e_ is None:
        return False
    s, e = ctx.segment(30, 0.6, persistent_prob=0.7)
    ramp = np.linspace(0, 1, e - s) ** 1.5
    for role, amount in [("optical_density", ctx.sd(ctx.col("optical_density", e_)) * 4 + 0.5),
                         ("dissolved_oxygen", -ctx.rng.uniform(10, 30)), ("ph", -ctx.rng.uniform(0.1, 0.3))]:
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] += amount * ctx.f * ramp
            ctx.mark(s, e, [c], "drift")
    return True


def sc_temp_controller_fault(ctx):
    c = ctx.pick(ctx.cols("culture_temp"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.5)
    ctx.X[s:e, c] += ctx.rng.choice([-1, 1]) * ctx.rng.uniform(1, 4) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_ph_probe_fouling(ctx):
    c = ctx.pick(ctx.cols("ph"))
    if c is None:
        return False
    s, e = ctx.segment(30, 0.6, persistent_prob=0.6)
    ctx.X[s:e, c] += ctx.rng.choice([-1, 1]) * ctx.rng.uniform(0.1, 0.4) * ctx.f * np.linspace(0, 1, e - s)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_stirrer_failure(ctx):
    e_ = ctx.pick_entity("stirrer_rpm")
    if e_ is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.4)
    cs, cd = ctx.col("stirrer_rpm", e_), ctx.col("dissolved_oxygen", e_)
    ctx.X[s:e, cs] = 0
    ctx.mark(s, e, [cs], "level_shift")
    if cd is not None:
        ctx.X[s:e, cd] = np.maximum(ctx.X[s:e, cd] - ctx.rng.uniform(20, 40) * ctx.rise(e - s), 0)
        ctx.mark(s, e, [cd], "drift")
    return True


# =============================================================================
# 8. ENERJİ: güneş santrali, rüzgâr türbini, şebeke yükü
# =============================================================================
def gen_energy(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h, doy, wd = _hours(t), _doy(t), _weekday(t)
    amb = 12 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + 6 * np.sin(2 * np.pi * (h - 9) / 24) \
        + _ar1(rng, T, 6 * 3600 / dt, 1.5)
    daylen = 12 + 3.5 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    elev = np.clip(np.sin(np.pi * (h - (12 - daylen / 2)) / daylen), 0, None)   # gece sıfır: NORMAL
    cols, e = _Cols(), 0
    while len(cols) < k:
        kind = rng.integers(3)
        if kind == 0:   # güneş
            cloud = 1 - 0.75 / (1 + np.exp(-(2 * _ar1(rng, T, 2700 / dt, 1.0) + rng.uniform(-1.5, 1.5)) * 2))
            irr = np.round(np.maximum(1000 * elev ** 1.2 * cloud + rng.normal(0, 3, T), 0), 1)
            mt = np.round(amb + 0.03 * irr + rng.normal(0, 0.3, T), 1)
            cap = 10 ** rng.uniform(1, 4.5)
            pv = np.round(np.clip(irr / 1000 * cap * (1 - 0.004 * (mt - 25)) * rng.uniform(0.95, 1.0)
                                  + rng.normal(0, 0.002 * cap, T), 0, cap * rng.uniform(0.8, 1.0)), 2)
            for x, role in [(irr, "irradiance"), (pv, "pv_power"), (mt, "module_temp")]:
                cols.add(x, role, e)
        elif kind == 1:  # rüzgâr
            ws = np.exp(_ar1(rng, T, 3 * 3600 / dt, 0.4)) * rng.uniform(4.5, 9) \
                * (1 + 0.1 * np.sin(2 * np.pi * (h - 15) / 24)) + np.abs(rng.normal(0, 0.3, T))
            rated = 10 ** rng.uniform(2.5, 3.8)
            p = np.where(ws < 3, 0, np.where(ws < 12, rated * ((ws - 3) / 9) ** 3, np.where(ws < 25, rated, 0)))
            p = np.round(np.maximum(p * (1 + rng.normal(0, 0.02, T)), 0), 1)
            rpm = np.round(np.where(p > 0, np.clip(ws * 1.2, 6, 15), rng.uniform(0, 1.5, T)), 2)
            for x, role in [(np.round(ws, 2), "wind_speed"), (p, "wind_power"), (rpm, "rotor_rpm")]:
                cols.add(x, role, e)
        else:            # şebeke yükü
            load = 10 ** rng.uniform(1, 4) * (0.6 + 0.25 * _bump(h, 9, 2.5) + 0.35 * _bump(h, 19.5, 2.5)) \
                * np.where(wd >= 5, 0.88, 1.0) * (1 + 0.004 * np.abs(amb - 18)) * np.exp(_ar1(rng, T, 3600 / dt, 0.03))
            cols.add(np.round(load, 1), "grid_load", e)
        e += 1
    return cols.out(k)


def sc_inverter_trip(ctx):
    """İnvertör devre dışı: güneş varken üretim sıfır."""
    e_ = ctx.pick_entity("pv_power", "irradiance")
    if e_ is None:
        return False
    cp, ci = ctx.col("pv_power", e_), ctx.col("irradiance", e_)
    day = np.where(ctx.X[:, ci] > 100)[0]
    day = day[(day >= ctx.T // 10) & (day < ctx.T - 3)]
    if len(day) < 3:
        return False
    s = int(ctx.rng.choice(day))
    night = np.where(ctx.X[s:, ci] <= 100)[0]
    e = min(s + int(ctx.rng.integers(3, max(4, ctx.T // 5))), s + night[0] if len(night) else ctx.T)
    ctx.X[s:e, cp] = 0
    ctx.mark(s, e, [cp], "level_shift")
    return True


def sc_soiling(ctx):
    """Panel kirlenmesi: verim yavaş yavaş düşer."""
    c = ctx.pick(ctx.cols("pv_power"))
    if c is None:
        return False
    s, e = ctx.segment(30, 0.6, persistent_prob=0.8)
    ctx.X[s:e, c] *= 1 - ctx.rng.uniform(0.1, 0.3) * ctx.f * np.linspace(0, 1, e - s)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_curtailment(ctx):
    """Şebeke kısıtı: üretim yapay bir tavanda kesilir."""
    c = ctx.pick(ctx.cols("pv_power") + ctx.cols("wind_power"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5)
    cap = ctx.rng.uniform(0.3, 0.6) * ctx.X[:, c].max()
    rows = s + np.where(ctx.X[s:e, c] > cap)[0]
    if len(rows) == 0:
        return False
    ctx.X[rows, c] = cap
    ctx.mark_rows(rows, [c], "pattern_change")
    return True


def sc_anemometer_fault(ctx):
    """Rüzgâr ölçer dondu; türbin üretmeye devam ediyor."""
    c = ctx.pick(ctx.cols("wind_speed"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.4)
    ctx.X[s:e, c] = ctx.X[s, c] if ctx.rng.random() < 0.6 else 0.0
    ctx.mark(s, e, [c], "flatline")
    return True


def sc_yaw_misalignment(ctx):
    """Türbin rüzgâra yanlış bakıyor: aynı rüzgârda daha az güç."""
    c = ctx.pick(ctx.cols("wind_power"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.4)
    ctx.X[s:e, c] *= 1 - ctx.rng.uniform(0.15, 0.4) * ctx.f
    ctx.mark(s, e, [c], "correlation_break")
    return True


def sc_blackout(ctx):
    c = ctx.pick(ctx.cols("grid_load"))
    if c is None:
        return False
    s, e = ctx.segment(3, 0.2)
    ctx.X[s:e, c] *= ctx.rng.uniform(0.02, 0.2)
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 9. BİLİŞİM: sunucu ve uygulama izleme
# =============================================================================
def gen_it(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h, wd = _hours(t), _weekday(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        req = 10 ** rng.uniform(0, 3.5) * (0.35 + 0.6 * _bump(h, 11, 3) + 0.5 * _bump(h, 20, 3)) \
            * np.where(wd >= 5, 0.7, 1.0) * np.exp(_ar1(rng, T, 600 / dt, 0.08))
        req = np.round(np.maximum(req * (1 + rng.normal(0, 0.05, T)), 0), 1)
        cpu = np.round(np.clip(rng.uniform(3, 10) + rng.uniform(40, 75) * req / (req.max() + 1e-9)
                               + rng.normal(0, 2, T), 0, 100), 1)
        lat = np.round(rng.uniform(5, 80) / (1 - np.minimum(cpu / 100, 0.95)) * np.exp(rng.normal(0, 0.1, T)), 1)
        errs = rng.poisson(np.maximum(req * dt * rng.uniform(1e-4, 5e-3), 0)).astype(float)
        m0 = rng.uniform(20, 50)
        mem = m0 + rng.uniform(10, 30) * (((t / rng.uniform(1200, 3 * 3600)) + rng.uniform()) % 1)  # GC testere dişi: NORMAL
        mem = np.round(np.clip(mem + rng.normal(0, 0.5, T), 0, 100), 1)
        for x, role in [(req, "requests"), (cpu, "cpu"), (lat, "latency"), (errs, "errors"), (mem, "memory")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_memory_leak(ctx):
    c = ctx.pick(ctx.cols("memory"))
    if c is None:
        return False
    s, e = ctx.segment(30, 0.6, persistent_prob=0.7)
    ctx.X[s:e, c] = np.clip(ctx.X[s:e, c] + ctx.rng.uniform(10, 40) * ctx.f * np.linspace(0, 1, e - s), 0, 100)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_outage(ctx):
    """Servis çöktü: istek yok, CPU boşta, başlangıçta hata patlaması."""
    e_ = ctx.pick_entity("requests")
    if e_ is None:
        return False
    s, e = ctx.segment(5, 0.25, persistent_prob=0.2)
    for role, val in [("requests", 0.0), ("cpu", ctx.rng.uniform(1, 3)), ("latency", 0.0)]:
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] = val
            ctx.mark(s, e, [c], "level_shift")
    c = ctx.col("errors", e_)
    if c is not None:
        e2 = min(e, s + 3)
        ctx.X[s:e2, c] += ctx.X[:, c].max() * ctx.rng.uniform(3, 10) + 10
        ctx.mark(s, e2, [c], "spike")
    return True


def sc_latency_regression(ctx):
    """Hatalı bir sürüm sonrası gecikme kalıcı olarak artar."""
    c = ctx.pick(ctx.cols("latency"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.5, persistent_prob=0.5)
    ctx.X[s:e, c] *= 1 + ctx.rng.uniform(0.5, 3) * ctx.f
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_error_burst(ctx):
    c = ctx.pick(ctx.cols("errors"))
    if c is None:
        return False
    s, e = ctx.segment(5, 0.25)
    ctx.X[s:e, c] += ctx.rng.poisson(max(ctx.X[:, c].mean(), 1) * ctx.rng.uniform(3, 15) * ctx.f, e - s)
    ctx.mark(s, e, [c], "noise_burst")
    return True


def sc_traffic_surge(ctx):
    """Ani trafik patlaması (kampanya, bot, DDoS)."""
    e_ = ctx.pick_entity("requests")
    if e_ is None:
        return False
    s, e = ctx.segment(3, 0.08)
    b = ctx.bell(e - s)
    for role, mult, cap in [("requests", ctx.rng.uniform(3, 8), None), ("cpu", ctx.rng.uniform(1.5, 2.5), 100),
                            ("latency", ctx.rng.uniform(2, 5), None)]:
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] *= 1 + (mult - 1) * ctx.f * b
            if cap:
                ctx.X[s:e, c] = np.minimum(ctx.X[s:e, c], cap)
            ctx.mark(s, e, [c], "spike")
    return True


# =============================================================================
# 10. SU ŞEBEKESİ: debi, basınç, depo seviyesi
# =============================================================================
def gen_water(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h = _hours(t)
    info_all = extra.setdefault("water", {})
    cols, e = _Cols(), 0
    while len(cols) < k:
        flow = 10 ** rng.uniform(0.5, 2.5) * (0.3 + 0.8 * _bump(h, 7.5, 1.3) + 0.35 * _bump(h, 13, 2)
                                             + 0.6 * _bump(h, 20, 2)) * np.exp(_ar1(rng, T, 1800 / dt, 0.05))
        flow = np.round(np.maximum(flow * (1 + rng.normal(0, 0.02, T)), 0), 2)
        pres = np.round(rng.uniform(3, 6) - rng.uniform(0.5, 1.5) * (flow / flow.max()) ** 2
                        + rng.normal(0, 0.02, T), 3)
        qin = 1.6 * flow.max()
        area = (qin - flow.mean()) * rng.uniform(2, 6) * 3600 / 60
        level, lv, pump = np.empty(T), rng.uniform(35, 85), rng.random() < 0.5
        for i in range(T):                          # pompa histerezisi: testere dişi NORMAL
            if lv < 30:
                pump = True
            elif lv > 90:
                pump = False
            lv = min(max(lv + ((qin if pump else 0) - flow[i]) * dt / area, 0), 100)
            level[i] = lv
        info_all[e] = {"area": area, "mean_flow": flow.mean()}
        for x, role in [(flow, "flow"), (pres, "pressure"), (np.round(level + rng.normal(0, 0.1, T), 1), "tank_level")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_leak(ctx):
    """Kaçak: özellikle gece minimum debisi yükselir, basınç hafifçe düşer."""
    e_ = ctx.pick_entity("flow")
    if e_ is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.7)
    cf, cp = ctx.col("flow", e_), ctx.col("pressure", e_)
    ctx.X[s:e, cf] += ctx.X[:, cf].mean() * ctx.rng.uniform(0.1, 0.3) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [cf], "level_shift")
    if cp is not None:
        ctx.X[s:e, cp] -= ctx.sd(cp) * ctx.rng.uniform(1, 3) * ctx.f * ctx.rise(e - s)
        ctx.mark(s, e, [cp], "level_shift")
    return True


def sc_pipe_burst(ctx):
    e_ = ctx.pick_entity("flow")
    if e_ is None:
        return False
    s, e = ctx.segment(3, 0.1)
    b = ctx.bell(e - s)
    cf, cp = ctx.col("flow", e_), ctx.col("pressure", e_)
    ctx.X[s:e, cf] += ctx.X[:, cf].max() * ctx.rng.uniform(0.5, 2) * ctx.f * b
    ctx.mark(s, e, [cf], "spike")
    if cp is not None:
        ctx.X[s:e, cp] -= ctx.rng.uniform(0.5, 2) * ctx.f * b
        ctx.mark(s, e, [cp], "spike")
    return True


def sc_pump_failure(ctx):
    """Pompa arızası: depo dolmaz, seviye sürekli düşer."""
    e_ = ctx.pick_entity("tank_level")
    if e_ is None or e_ not in ctx.extra.get("water", {}):
        return False
    info = ctx.extra["water"][e_]
    c = ctx.col("tank_level", e_)
    s, e = ctx.segment(20, 0.5, persistent_prob=0.6)
    decline = info["mean_flow"] * ctx.dt / info["area"] * ctx.rng.uniform(0.8, 1.2)
    ctx.X[s:e, c] = np.clip(ctx.X[s, c] - decline * np.arange(1, e - s + 1), 0, 100)
    ctx.mark(s, e, [c], "drift")
    return True


# =============================================================================
# 11. ÇEVRE / İKLİM: meteoroloji ve hava kalitesi istasyonları
# =============================================================================
def gen_environment(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h, doy = _hours(t), _doy(t)
    syn = _ar1(rng, T, 2 * 86400 / dt, 1.0)      # istasyonlar arası ortak hava sistemi
    syn2 = _ar1(rng, T, 3 * 86400 / dt, 1.0)
    rain_drive = _ar1(rng, T, 2 * 3600 / dt, 1.0)
    cols, e = _Cols(), 0
    while len(cols) < k:
        temp = 14 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.uniform(4, 7) * np.sin(2 * np.pi * (h - 9) / 24) \
            + 3 * syn + _ar1(rng, T, 3 * 3600 / dt, 0.6) + rng.normal(0, 2)
        raining = (rain_drive + 0.3 * _ar1(rng, T, 3600 / dt, 1.0)) > rng.uniform(1.0, 1.8)
        rain = np.round(np.where(raining, rng.gamma(1.5, 0.8, T) * dt / 3600, 0), 1)
        hum = np.round(np.clip(70 - 2.2 * (temp - temp.mean()) + _ar1(rng, T, 6 * 3600 / dt, 5) + 15 * raining, 5, 100))
        pres = np.round(1013 + 8 * syn2 + rng.normal(0, 0.15, T) + rng.normal(0, 3), 1)
        pm = np.round(rng.uniform(8, 30) * np.exp(_ar1(rng, T, 8 * 3600 / dt, 0.45))
                      * (1 + 0.5 * _bump(h, 8, 1.5) + 0.4 * _bump(h, 19, 2)), 1)
        for x, role in [(np.round(temp, 1), "air_temp"), (hum, "humidity"), (pres, "air_pressure"),
                        (pm, "pm25"), (rain, "rain")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_calibration_drift(ctx):
    c = ctx.pick(ctx.cols("air_temp") + ctx.cols("humidity") + ctx.cols("air_pressure"))
    if c is None:
        return False
    s, e = ctx.segment(30, 0.6, persistent_prob=0.6)
    ctx.X[s:e, c] += ctx.rng.choice([-1, 1]) * ctx.sd(c) * ctx.rng.uniform(2, 5) * ctx.f * np.linspace(0, 1, e - s)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_humidity_stuck(ctx):
    c = ctx.pick(ctx.cols("humidity"))
    if c is None:
        return False
    s, e = ctx.segment(15, 0.5, persistent_prob=0.4)
    ctx.X[s:e, c] = 100.0
    ctx.mark(s, e, [c], "flatline")
    return True


def sc_radiation_shield_error(ctx):
    """Radyasyon kalkanı bozuk: öğle saatlerinde sahte sıcaklık artışı."""
    c = ctx.pick(ctx.cols("air_temp"))
    if c is None:
        return False
    s, e = ctx.segment(30, 0.7, persistent_prob=0.5)
    add = ctx.rng.uniform(2, 6) * ctx.f * _bump(_hours(ctx.t[s:e]), 13, 2)
    rows = s + np.where(add > 0.3)[0]
    if len(rows) == 0:
        return False
    ctx.X[s:e, c] += add
    ctx.mark_rows(rows, [c], "pattern_change")
    return True


def sc_pollution_episode(ctx):
    c = ctx.pick(ctx.cols("pm25"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4)
    ctx.X[s:e, c] *= 1 + ctx.rng.uniform(2, 7) * ctx.f * ctx.bell(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 12. PERAKENDE: ürün satışları ve fiyatlar
# =============================================================================
_WEEKLY = np.array([1.0, 0.95, 1.0, 1.05, 1.2, 1.5, 1.3])


def gen_retail(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h, wd = _hours(t), _weekday(t)
    hourly = dt < 86400
    cols, e = _Cols(), 0
    while len(cols) < k:
        base = 10 ** rng.uniform(-0.3, 2.3) * ((dt / 86400) * (24 / 13) if hourly else 1)
        shape = _WEEKLY[wd] * (np.where((h >= 9) & (h < 22), 0.6 + 0.8 * _bump(h, 18, 3), 0.0) if hourly else 1)
        p0 = np.round(10 ** rng.uniform(0, 2.7), 2)
        price = np.full(T, p0)
        if rng.random() < 0.4:                      # kampanya: fiyat düşer, satış artar (NORMAL)
            s = int(rng.integers(0, T - 5))
            price[s:s + int(rng.integers(3, max(4, T // 6)))] = np.round(p0 * rng.uniform(0.7, 0.9), 2)
        demand = base * shape * np.exp(_ar1(rng, T, 30 * 86400 / dt, 0.15)) * (price / p0) ** (-rng.uniform(1.5, 3))
        cols.add(rng.poisson(demand).astype(float), "sales", e)
        cols.add(price, "price", e)
        e += 1
    return cols.out(k)


def sc_stockout(ctx):
    """Stok bitti: satışlar sıfıra iner."""
    cands = [c for c in ctx.cols("sales") if ctx.X[:, c].mean() >= 1.5]
    c = ctx.pick(cands)
    if c is None:
        return False
    s, e = ctx.segment(5, 0.3, persistent_prob=0.3)
    ctx.X[s:e, c] = 0
    ctx.mark(s, e, [c], "flatline")
    return True


def sc_data_entry_error(ctx):
    c = ctx.pick(ctx.cols("sales") + ctx.cols("price"))
    if c is None:
        return False
    i = int(ctx.rng.integers(ctx.T // 10, ctx.T))
    ctx.X[i, c] = max(ctx.X[i, c], 1) * ctx.rng.choice([10, 100]) * ctx.rng.uniform(0.9, 1.1)
    ctx.mark(i, i + 1, [c], "spike")
    return True


def sc_demand_shift(ctx):
    c = ctx.pick(ctx.cols("sales"))
    if c is None:
        return False
    s, e = ctx.segment(15, 0.5, persistent_prob=0.6)
    mult = ctx.rng.uniform(1.8, 3) if ctx.rng.random() < 0.5 else ctx.rng.uniform(0.2, 0.5)
    ctx.X[s:e, c] = np.round(ctx.X[s:e, c] * (1 + (mult - 1) * ctx.f / 1.5))
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 13. OTOMOTİV: araç telemetrisi (CAN bus)
# =============================================================================
def gen_automotive(rng, t, k, extra):
    T, dt = len(t), _step(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        target, i = np.empty(T), 0
        while i < T:                                # dur-kalk, şehir içi, otoyol
            dur = int(rng.integers(max(2, int(20 / dt)), max(3, int(200 / dt)) + 1))
            target[i:i + dur] = rng.choice([0, 30, 50, 70, 90, 110, 130]) * rng.uniform(0.85, 1.1)
            i += dur
        speed = np.round(np.maximum(_lowpass(target, np.exp(-dt / 8)) + rng.normal(0, 0.3, T), 0), 1)
        ratios = np.array([110, 65, 45, 35, 29, 25]) * rng.uniform(0.9, 1.1)
        gear = np.searchsorted([15, 30, 50, 70, 90], speed)
        rpm = np.round(np.where(speed < 2, 800, np.maximum(speed * ratios[gear], 900)) + rng.normal(0, 15, T))
        amb = rng.uniform(-5, 35)
        cool = 90 - (90 - amb) * np.exp(-(t - t[0]) / rng.uniform(300, 700)) if rng.random() < 0.3 else np.full(T, 90.0)
        cool = np.round(cool + 1.5 * np.sin(2 * np.pi * t / rng.uniform(120, 300)) * (cool > 85)
                        + 0.01 * (rpm - 2000) / 100 + rng.normal(0, 0.2, T), 1)
        fuel = np.round(rng.uniform(20, 90) - np.cumsum(rpm) * dt * rng.uniform(1.5e-6, 3e-6) + rng.normal(0, 0.3, T), 1)
        batt = np.round(14.2 + rng.normal(0, 0.05, T) - 0.3 * (rpm < 900), 2)
        for x, role in [(speed, "speed"), (rpm, "engine_rpm"), (cool, "coolant_temp"),
                        (fuel, "fuel_level"), (batt, "battery_voltage")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_engine_overheat(ctx):
    c = ctx.pick(ctx.cols("coolant_temp"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.6)
    ctx.X[s:e, c] += ctx.rng.uniform(10, 25) * ctx.f * np.linspace(0, 1, e - s)
    ctx.mark(s, e, [c], "drift")
    return True


def sc_thermostat_stuck_open(ctx):
    c = ctx.pick(ctx.cols("coolant_temp"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.6)
    ctx.X[s:e, c] -= ctx.rng.uniform(10, 20) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_clutch_slip(ctx):
    """Debriyaj kaçırıyor: aynı hızda devir yüksek."""
    e_ = ctx.pick_entity("engine_rpm", "speed")
    if e_ is None:
        return False
    cr, cs = ctx.col("engine_rpm", e_), ctx.col("speed", e_)
    s, e = ctx.segment(10, 0.4, persistent_prob=0.3)
    rows = s + np.where(ctx.X[s:e, cs] > 20)[0]
    if len(rows) < 3:
        return False
    ctx.X[rows, cr] *= 1 + ctx.rng.uniform(0.2, 0.5) * ctx.f
    ctx.mark_rows(rows, [cr], "correlation_break")
    return True


def sc_fuel_theft(ctx):
    """Yakıt hırsızlığı: seviye kısa sürede sert düşer."""
    c = ctx.pick(ctx.cols("fuel_level"))
    if c is None:
        return False
    s, e = ctx.segment(3, 0.05)
    drop = ctx.rng.uniform(10, 30) * ctx.f
    ctx.X[s:e, c] -= drop * np.linspace(0, 1, e - s)
    ctx.X[e:, c] -= drop
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_alternator_failure(ctx):
    c = ctx.pick(ctx.cols("battery_voltage"))
    if c is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.7)
    ctx.X[s:e, c] -= ctx.rng.uniform(0.8, 2) * ctx.f * np.linspace(0, 1, e - s)
    ctx.mark(s, e, [c], "drift")
    return True


# =============================================================================
# 14. TELEKOM: baz istasyonu hücreleri
# =============================================================================
def gen_telecom(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h = _hours(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        users = rng.poisson(10 ** rng.uniform(1.5, 3.5) * (0.15 + 0.5 * _bump(h, 12, 3) + 0.8 * _bump(h, 21, 2.5))
                            * np.exp(_ar1(rng, T, 3600 / dt, 0.05))).astype(float)
        traffic = np.round(users * rng.uniform(0.02, 0.1) * (dt / 900) * np.exp(rng.normal(0, 0.1, T)), 3)
        prb = np.round(np.clip(100 * traffic / (traffic.max() + 1e-9) * rng.uniform(0.55, 0.9)
                               + rng.normal(0, 2, T), 0, 100), 1)
        drop = np.round(np.clip(0.3 + 2.5 * (prb / 100) ** 4 + rng.normal(0, 0.05, T), 0, 100), 2)
        for x, role in [(users, "users"), (traffic, "traffic"), (prb, "prb_util"), (drop, "drop_rate")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_cell_outage(ctx):
    e_ = ctx.pick_entity("users")
    if e_ is None:
        return False
    s, e = ctx.segment(3, 0.25, persistent_prob=0.2)
    cs = [c for c in ctx.cols(entity=e_)]
    ctx.X[s:e, cs] = 0
    ctx.mark(s, e, cs, "level_shift")
    return True


def sc_sleeping_cell(ctx):
    """Uyuyan hücre: kullanıcılar bağlı görünüyor ama trafik taşınmıyor."""
    e_ = ctx.pick_entity("traffic")
    if e_ is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.5)
    for role, m in [("traffic", 0.05), ("prb_util", 0.1)]:
        c = ctx.col(role, e_)
        if c is not None:
            ctx.X[s:e, c] *= m
            ctx.mark(s, e, [c], "correlation_break")
    return True


def sc_congestion(ctx):
    e_ = ctx.pick_entity("prb_util")
    if e_ is None:
        return False
    s, e = ctx.segment(5, 0.3)
    cp, cd = ctx.col("prb_util", e_), ctx.col("drop_rate", e_)
    ctx.X[s:e, cp] = 100.0
    ctx.mark(s, e, [cp], "flatline")
    if cd is not None:
        ctx.X[s:e, cd] += ctx.rng.uniform(2, 8) * ctx.f
        ctx.mark(s, e, [cd], "level_shift")
    return True


# =============================================================================
# 15. TARIM: toprak nemi, sulama, sera
# =============================================================================
def gen_agriculture(rng, t, k, extra):
    T, dt = len(t), _step(t)
    h = _hours(t)
    daylight = np.clip(np.sin(np.pi * (h - 6) / 13), 0, None)
    info_all = extra.setdefault("agri", {})
    schedules = [[6.0], [6.0, 18.0], [5.0]]
    cols, e = _Cols(), 0
    while len(cols) < k:
        loss = rng.uniform(0.15, 0.5) * (0.15 + daylight) * dt / 3600      # buharlaşma (%/adım)
        hours_ = schedules[int(rng.integers(3))]
        every = int(rng.choice([1, 1, 2]))
        amount = rng.uniform(6, 14)
        m, mv, pending = np.empty(T), rng.uniform(20, 40), 0.0
        for i in range(T):                          # sulama sıçramaları NORMALDİR
            day = int(t[i] // 86400)
            if i > 0 and day % every == 0:
                for ih in hours_:
                    if t[i - 1] < day * 86400 + ih * 3600 <= t[i]:
                        pending += amount
            add = pending if dt >= 1800 else min(pending, amount * dt / 1800)
            pending -= add
            mv = min(max(mv + add - loss[i], 5), 55)
            m[i] = mv
        soil_t = np.round(18 + 3 * np.sin(2 * np.pi * (h - 15) / 24) + _ar1(rng, T, 6 * 3600 / dt, 0.5) + rng.normal(0, 1), 1)
        co2 = np.round(420 + 220 * _bump(h, 3, 4) - 90 * daylight + _ar1(rng, T, 1800 / dt, 15))
        info_all[e] = {"loss": loss, "amount": amount}
        for x, role in [(np.round(m + rng.normal(0, 0.2, T), 1), "soil_moisture"),
                        (soil_t, "soil_temp"), (co2, "greenhouse_co2")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_irrigation_failure(ctx):
    """Sulama çalışmadı: nem beklenen sıçramalar olmadan düşmeye devam eder."""
    e_ = ctx.pick_entity("soil_moisture")
    if e_ is None or e_ not in ctx.extra.get("agri", {}):
        return False
    c = ctx.col("soil_moisture", e_)
    min_len = min(int(86400 / ctx.dt), ctx.T // 2)
    s, e = ctx.segment(max(min_len, 5), 0.7, persistent_prob=0.7)
    loss = ctx.align(ctx.extra["agri"][e_]["loss"])
    new = np.clip(ctx.X[s, c] - np.cumsum(loss[s:e]), 5, 55)
    if np.abs(new - ctx.X[s:e, c]).max() < 1.0:      # bu aralıkta sulama yoktu, fark oluşmadı
        return False
    ctx.X[s:e, c] = new
    ctx.mark(s, e, [c], "drift")
    return True


def sc_valve_stuck_open(ctx):
    c = ctx.pick(ctx.cols("soil_moisture"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.5, persistent_prob=0.5)
    ctx.X[s:e, c] = np.minimum(ctx.X[s, c] + np.cumsum(np.full(e - s, 20 * ctx.dt / 3600)), 55)
    ctx.mark(s, e, [c], "level_shift")
    return True


def sc_co2_fault(ctx):
    c = ctx.pick(ctx.cols("greenhouse_co2"))
    if c is None:
        return False
    s, e = ctx.segment(10, 0.4, persistent_prob=0.4)
    ctx.X[s:e, c] += ctx.rng.uniform(200, 600) * ctx.f * ctx.rise(e - s)
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 16. KİMYASAL PROSES: reaktör kontrol döngüsü
# =============================================================================
def gen_chemical(rng, t, k, extra):
    T, dt = len(t), _step(t)
    cols, e = _Cols(), 0
    while len(cols) < k:
        sp = np.full(T, rng.uniform(120, 250))
        for _ in range(int(rng.integers(0, 4))):     # set noktası değişimleri NORMALDİR
            sp[int(rng.integers(T // 10, T)):] = rng.uniform(120, 250)
        zeta, Ts = rng.uniform(0.35, 0.8), rng.uniform(120, 900)
        w = 4 / (zeta * Ts)
        lam, V = np.linalg.eig(np.array([[0.0, 1.0], [-w * w, -2 * zeta * w]]) * dt)
        Phi = (V @ np.diag(np.exp(lam)) @ np.linalg.inv(V)).real     # 2. derece sistemin tam ayrıklaştırması
        state, temp = np.zeros(2), np.empty(T)
        for i in range(T):
            if i > 0 and sp[i] != sp[i - 1]:
                state[0] -= sp[i] - sp[i - 1]
            state = Phi @ state
            temp[i] = sp[i] + state[0]
        temp = np.round(temp + rng.normal(0, 0.15, T), 2)
        pres = np.round(1 + 0.012 * (temp - 20) * rng.uniform(0.8, 1.2) + rng.normal(0, 0.01, T), 3)
        flow = np.round(rng.uniform(1, 20) * (1 + 0.3 * (sp - 185) / 65) * (1 + rng.normal(0, 0.01, T)), 2)
        valve = np.round(np.clip(45 + 25 * (sp - 185) / 65 + 1.5 * (sp - temp) + rng.normal(0, 0.8, T), 0, 100), 1)
        for x, role in [(temp, "reactor_temp"), (pres, "reactor_pressure"), (flow, "feed_flow"),
                        (valve, "valve_position")]:
            cols.add(x, role, e)
        e += 1
    return cols.out(k)


def sc_thermal_runaway(ctx):
    e_ = ctx.pick_entity("reactor_temp")
    if e_ is None:
        return False
    s, e = ctx.segment(15, 0.4, persistent_prob=0.6)
    r = np.linspace(0, 1, e - s)
    grow = ctx.rng.uniform(5, 30) * ctx.f * (np.exp(3 * r) - 1) / (np.e ** 3 - 1)
    ct, cp = ctx.col("reactor_temp", e_), ctx.col("reactor_pressure", e_)
    ctx.X[s:e, ct] += grow
    ctx.mark(s, e, [ct], "drift")
    if cp is not None:
        ctx.X[s:e, cp] += 0.012 * grow
        ctx.mark(s, e, [cp], "drift")
    return True


def sc_valve_stiction(ctx):
    """Vana yapışması: vana basamaklı hareket eder, sıcaklık salınır."""
    e_ = ctx.pick_entity("valve_position")
    if e_ is None:
        return False
    s, e = ctx.segment(20, 0.5, persistent_prob=0.4)
    cv, ct = ctx.col("valve_position", e_), ctx.col("reactor_temp", e_)
    q = ctx.rng.uniform(3, 8)
    ctx.X[s:e, cv] = np.round(ctx.X[s:e, cv] / q) * q
    ctx.mark(s, e, [cv], "pattern_change")
    if ct is not None:
        ctx.X[s:e, ct] += ctx.rng.uniform(1, 4) * ctx.f * np.sin(2 * np.pi * np.arange(e - s) / ctx.rng.uniform(6, 20))
        ctx.mark(s, e, [ct], "pattern_change")
    return True


def sc_feed_pump_trip(ctx):
    c = ctx.pick(ctx.cols("feed_flow"))
    if c is None:
        return False
    s, e = ctx.segment(5, 0.3, persistent_prob=0.3)
    ctx.X[s:e, c] = 0
    ctx.mark(s, e, [c], "level_shift")
    return True


# =============================================================================
# 17. SOYUT SERİLER: hiçbir alana ait olmayan genel davranışlar
# =============================================================================
def gen_abstract(rng, t, k, extra):
    T, dt = len(t), _step(t)
    spd = 86400 / dt
    def seasonal():
        tt = np.arange(T)
        x = np.zeros(T)
        for p in rng.choice([spd, spd * 7, rng.integers(8, 200)], size=int(rng.integers(1, 3)), replace=False):
            x += rng.uniform(0.3, 2) * np.sin(2 * np.pi * tt / max(float(p), 4.0) + rng.uniform(0, 6.3))
        return x + rng.normal(0, rng.uniform(0.05, 0.4), T) + rng.uniform(-0.002, 0.002) * tt
    def walk():
        return np.cumsum(rng.standard_t(rng.uniform(3, 8), T) * np.exp(_ar1(rng, T, 20, 0.3)))
    def intermittent():
        return (rng.random(T) < rng.uniform(0.05, 0.4)) * rng.gamma(2.0, rng.uniform(1, 10), T)
    def regime():
        lv = rng.normal(0, 3, int(rng.integers(2, 4)))
        idx = np.cumsum(rng.random(T) < 0.005) % len(lv)
        return lv[idx] + rng.normal(0, 0.2, T)
    gens = [seasonal, walk, intermittent, regime]
    n_lat = int(rng.integers(1, min(k, 5) + 1))
    lat = np.stack([gens[i]() for i in rng.integers(4, size=n_lat)], axis=1)
    lat = np.clip(np.nan_to_num(lat, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)   # taşma koruması
    lat = (lat - lat.mean(0)) / (lat.std(0) + 1e-8)
    W = rng.normal(0, 1, (n_lat, k)) * (rng.random((n_lat, k)) < 0.6)
    X = lat @ W + rng.normal(0, rng.uniform(0.05, 0.5), (T, k))
    scale = 10 ** rng.uniform(-2, 4, k)
    X = X * scale + rng.normal(0, 1, k) * scale * 5
    return X, ["abstract"] * k, np.zeros(k, dtype=int)


# =============================================================================
# 18. BAĞLAM-BAĞIMLI ÇOK DEĞİŞKENLİ SERİLER (TimeRCD tarzı)
#   Kaynak sinyaller (trend + rastgele dalga biçimli mevsimsellik + gürültü) rastgele bir DAG üzerinde
#   gecikmeli ARX dinamiğiyle birbirine bağlanır; gözlenen sütunlar bu kaynakların karışımıdır.
#   Anomali ENDOJEN (kaynağa, karışımdan önce → bağımlı sütunlara yayılır, etiket de yayılır) ya da
#   EKSOJEN (gözleme doğrudan) enjekte edilir. Amaç: aynı biçim bir bağlamda normal, başkasında anomali
#   olabilsin; model biçimi değil bağlamla uyuşmazlığı öğrensin.
# =============================================================================
def _waveform(rng, phase):
    kind = rng.choice(["sin", "square", "tri", "saw", "pulse", "wavelet"])
    if kind == "sin":
        return np.sin(phase)
    if kind == "square":
        return np.sign(np.sin(phase))
    if kind == "tri":
        return 2 * np.abs(2 * (phase / (2 * np.pi) % 1) - 1) - 1
    if kind == "saw":
        return 2 * (phase / (2 * np.pi) % 1) - 1
    if kind == "pulse":
        return (np.sin(phase) > rng.uniform(0.5, 0.95)).astype(float) * 2 - 1
    return np.sin(phase) * np.exp(-((phase / (2 * np.pi) % 1) - 0.5) ** 2 / rng.uniform(0.01, 0.1))


def _coupled_source(rng, T):
    tt = np.arange(T, dtype=float)
    x = np.zeros(T)
    for _ in range(int(rng.integers(0, 3))):                          # 0–2 mevsimsel bileşen
        period = float(rng.choice([rng.uniform(6, 40), rng.uniform(40, max(41, T / 3)), rng.uniform(max(41, T / 3), max(42, T / 1.5))]))
        x += rng.uniform(0.3, 1.5) * _waveform(rng, 2 * np.pi * tt / period + rng.uniform(0, 6.3))
    if rng.random() < 0.6:                                             # trend: doğrusal / parçalı / rastgele yürüyüş
        kind = rng.choice(["lin", "pw", "rw"])
        if kind == "lin":
            x += rng.uniform(-1.5, 1.5) * tt / T
        elif kind == "pw":
            x += np.cumsum(rng.normal(0, 1, T) * (rng.random(T) < 0.01)) * rng.uniform(0.2, 0.8)
        else:
            x += np.cumsum(rng.normal(0, rng.uniform(0.01, 0.08), T))
    if rng.random() < 0.3:                                             # açma-kapama (duty cycle): kompresör, pompa, HVAC — NORMAL davranış
        period = rng.uniform(10, max(12.0, T / 8))
        duty = rng.uniform(0.15, 0.85)
        jitter = np.cumsum(rng.normal(0, 0.02, T))                      # periyot hafif kayar (gerçek makine)
        on = ((tt / period + jitter) % 1.0) < duty
        x += rng.uniform(1.0, 3.0) * on.astype(float)
    x += rng.normal(0, rng.uniform(0.02, 0.4), T)
    if rng.random() < 0.2:                                             # rejim değişimi: normal ama tuhaf
        a = int(rng.integers(T // 4, 3 * T // 4)); x[a:] += rng.normal(0, 1.5)
    return x


def gen_coupled(rng, t, k, extra):
    T = len(t)
    n_src = int(rng.integers(1, min(6, k) + 1))
    S = np.column_stack([_coupled_source(rng, T) for _ in range(n_src)])
    # DAG: kaynak j, i<j kaynaklarından gecikmeli ARX ile etkilenir
    A = np.zeros((n_src, n_src)); D = np.zeros((n_src, n_src), dtype=int)
    for j in range(1, n_src):
        for i in range(j):
            if rng.random() < 0.5:
                A[i, j] = rng.uniform(-0.8, 0.8); D[i, j] = int(rng.integers(0, 20))
    ar = rng.uniform(-0.3, 0.8, n_src)                                  # otoregresif katsayı |a| ≤ 0.8
    def propagate(S0):
        Z = S0.copy()
        for j in range(n_src):
            drive = Z[:, j].copy()
            for i in range(j):
                if A[i, j]:
                    d = D[i, j]
                    drive[d:] += A[i, j] * Z[:T - d, i]
            y = np.zeros(T)
            for n in range(T):                                          # y_n = a·y_{n-1} + drive_n
                y[n] = ar[j] * (y[n - 1] if n else 0.0) + drive[n]
            Z[:, j] = y
        return Z
    Z = propagate(S)
    # gözlem: her sütun 1–2 kaynağın doğrusal karışımı + sensör gürültüsü + ölçek/ofset
    W = np.zeros((n_src, k))
    for c in range(k):
        for i in rng.choice(n_src, size=min(n_src, int(rng.integers(1, 3))), replace=False):
            W[i, c] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    scale = 10 ** rng.uniform(-2, 3, k); offset = rng.normal(0, 2, k) * scale
    noise_sd = rng.uniform(0.02, 0.3, k)
    noise = rng.normal(0, 1, (T, k)) * noise_sd                           # sabit: yeniden gözlemde yalnızca kaynak değişikliği görünsün
    def observe(Zc):
        return (Zc @ W + noise) * scale + offset
    X = observe(Z)
    # torunlar: DAG üzerinde erişilebilirlik
    reach = {i: {i} for i in range(n_src)}
    for i in range(n_src):
        stack = [i]
        while stack:
            u = stack.pop()
            for v in range(u + 1, n_src):
                if A[u, v] and v not in reach[i]:
                    reach[i].add(v); stack.append(v)
    extra.update(S=S, W=W, reach=reach, propagate=propagate, observe=observe, D=D, n_src=n_src)
    return X, ["coupled"] * k, np.array([int(np.argmax(np.abs(W[:, c]))) for c in range(k)])


def sc_endogenous(ctx):
    """Kaynağa (karışımdan önce) enjeksiyon: bozulma bağımlı kaynaklara ve onlara bağlı sütunlara yayılır;
    etiket yayılan sütunlara (gecikme payıyla) yazılır."""
    ex = ctx.extra
    if "S" not in ex:
        return False
    rng, T = ctx.rng, ctx.T
    i = int(rng.integers(ex["n_src"]))
    S = ex["S"].copy()
    kind = str(rng.choice(["spike", "level_shift", "drift", "noise_burst", "flatline", "pattern_change"]))
    s, e = ctx.segment(5, 0.3, PERSISTENT_RATIO if kind in PERSISTENT_OK else 0.0)
    if kind == "spike":
        s = int(rng.integers(T // 10, T - 1)); e = min(T, s + int(rng.integers(1, 4)))
    L = e - s; sd = max(robust_std(S[:, i]), 1e-6); g = rng.uniform(2, 5) * ctx.f
    seg = slice(s, e)
    if kind == "spike":
        S[seg, i] += rng.choice([-1, 1]) * g * 1.5 * sd
    elif kind == "level_shift":
        S[seg, i] += rng.choice([-1, 1]) * g * 0.6 * sd
    elif kind == "drift":
        S[seg, i] += rng.choice([-1, 1]) * g * sd * np.linspace(0, 1, L)
    elif kind == "noise_burst":
        S[seg, i] += rng.normal(0, g * 0.5 * sd, L)
    elif kind == "flatline":
        S[seg, i] = S[s, i]
    else:
        S[seg, i] = np.median(S[seg, i]) + 1.5 * sd * np.sin(2 * np.pi * np.arange(L) / rng.uniform(3, max(4.0, L / 2)))
    Xn = ex["observe"](ex["propagate"](S))
    affected = ex["reach"][i]
    cols = [c for c in range(ctx.k) if any(ex["W"][j, c] for j in affected)]
    if not cols:
        return False
    delay = int(max([ex["D"][i, j] for j in affected] + [0]))
    e2 = min(T, e + delay + 1)
    # sütun bazında etki: değişim o sütunun oynaklığına göre anlamlıysa etiketlenir (zayıf yayılım etiketlenmez)
    eff = np.array([np.mean(np.abs(Xn[s:e2, c] - ctx.X[s:e2, c])) / max(robust_std(ctx.X[:, c]), 1e-9) for c in cols])
    lab_cols = [c for c, v in zip(cols, eff) if v >= 0.3]
    if not lab_cols:
        return False
    ctx.X[:, cols] = Xn[:, cols]
    ex["S"] = S
    ctx.mark(s, e2, lab_cols, kind if kind in TYPE_ID else "pattern_change")
    return True


def gen_real_template(rng, t, k, extra):
    """ŞABLON: kendi gerçek verinizden örnek çekmek için.
    Bir kaynaktan (fabrika, borsa, sunucu ...) len(t) satır ve k sütun alın,
    (T, k) matris döndürün. Rolleri bilmiyorsanız hepsine "unknown" verin;
    bu durumda sadece genel anomaliler uygulanır.
        X = kaynak_sec_ve_kes(rng, len(t), k)
        return X, ["unknown"] * k, np.zeros(k, dtype=int)
    """
    raise NotImplementedError


# =============================================================================
# Alan kaydı
# =============================================================================
DOMAINS = {
    "finance":       dict(gen=gen_finance, steps=[60, 300, 3600, 86400], business_days=True,
                          scenarios=[sc_flash_crash, sc_trading_halt, sc_fat_finger, sc_volatility_burst]),
    "manufacturing": dict(gen=gen_manufacturing, steps=[1, 10, 60, 300],
                          scenarios=[sc_bearing_wear, sc_unexpected_stop, sc_current_mismatch, sc_overheat]),
    "space":         dict(gen=gen_space, steps=[10, 30, 60],
                          scenarios=[sc_seu, sc_battery_degradation, sc_wheel_friction, sc_heater_stuck,
                                     sc_solar_string_failure]),
    "vitals":        dict(gen=gen_vitals, steps=[1, 5, 60],
                          scenarios=[sc_tachycardia, sc_desaturation, sc_fever, sc_probe_off,
                                     sc_motion_artifact, sc_hypotension]),
    "ecg":           dict(gen=gen_ecg, steps=[0.004, 0.008, 0.016], jitter=False,
                          scenarios=[sc_ectopic_beat, sc_ectopic_beat, sc_lead_off, sc_muscle_noise]),
    "glucose":       dict(gen=gen_cgm, steps=[60, 300, 900],
                          scenarios=[sc_hypoglycemia, sc_compression_low, sc_hyperglycemia]),
    "bioreactor":    dict(gen=gen_bioreactor, steps=[60, 300, 600],
                          scenarios=[sc_contamination, sc_temp_controller_fault, sc_ph_probe_fouling,
                                     sc_stirrer_failure]),
    "energy":        dict(gen=gen_energy, steps=[60, 300, 900, 3600],
                          scenarios=[sc_inverter_trip, sc_soiling, sc_curtailment, sc_anemometer_fault,
                                     sc_yaw_misalignment, sc_blackout]),
    "it":            dict(gen=gen_it, steps=[10, 60, 300],
                          scenarios=[sc_memory_leak, sc_outage, sc_latency_regression, sc_error_burst,
                                     sc_traffic_surge]),
    "water":         dict(gen=gen_water, steps=[60, 300, 900],
                          scenarios=[sc_leak, sc_pipe_burst, sc_pump_failure]),
    "environment":   dict(gen=gen_environment, steps=[300, 600, 3600],
                          scenarios=[sc_calibration_drift, sc_humidity_stuck, sc_radiation_shield_error,
                                     sc_pollution_episode]),
    "retail":        dict(gen=gen_retail, steps=[3600, 86400],
                          scenarios=[sc_stockout, sc_data_entry_error, sc_demand_shift]),
    "automotive":    dict(gen=gen_automotive, steps=[1, 5],
                          scenarios=[sc_engine_overheat, sc_thermostat_stuck_open, sc_clutch_slip,
                                     sc_fuel_theft, sc_alternator_failure]),
    "telecom":       dict(gen=gen_telecom, steps=[300, 900, 3600],
                          scenarios=[sc_cell_outage, sc_sleeping_cell, sc_congestion]),
    "agriculture":   dict(gen=gen_agriculture, steps=[300, 900, 3600],
                          scenarios=[sc_irrigation_failure, sc_valve_stuck_open, sc_co2_fault]),
    "chemical":      dict(gen=gen_chemical, steps=[1, 10, 60],
                          scenarios=[sc_thermal_runaway, sc_valve_stiction, sc_feed_pump_trip]),
    "abstract":      dict(gen=gen_abstract, steps=[1, 60, 300, 900, 3600, 86400], scenarios=[]),
    "coupled":       dict(gen=gen_coupled, steps=[1, 10, 60, 300, 900, 3600, 86400],
                          scenarios=[sc_endogenous, sc_endogenous, sc_endogenous]),   # endojen ağırlıklı; genel (eksojen) enjeksiyon make_sample'da
}
DOMAIN_NAMES = list(DOMAINS)
DOMAIN_WEIGHTS = np.ones(len(DOMAIN_NAMES)); DOMAIN_WEIGHTS[DOMAIN_NAMES.index("coupled")] = 0.4 * (len(DOMAIN_NAMES) - 1) / 0.6
DOMAIN_WEIGHTS = DOMAIN_WEIGHTS / DOMAIN_WEIGHTS.sum()             # coupled %40, kalan 17 alan eşit paylaşır


# =============================================================================
# Zaman damgaları
# =============================================================================
def make_timestamps(rng, n, step, business_days=False, jitter=True):
    start = 1.42e9 + rng.uniform(0, 3.5e8)                 # 2015-2026 arası
    if business_days and step >= 86400:                   # borsa: hafta sonu boşlukları NORMAL
        day0 = int(start // 86400)
        days = np.arange(day0, day0 + int(n * 1.5) + 10)
        days = days[(days + 3) % 7 < 5][:n]
        return days * 86400.0 + 57600.0                    # kapanış 16:00
    start = np.floor(start / step) * step
    dt = np.full(n, float(step))
    if jitter and rng.random() < 0.2:                      # düzensiz örnekleme
        dt *= rng.uniform(0.95, 1.05, n)
    dt[0] = 0.0
    return start + np.cumsum(dt)


# =============================================================================
# Ön işleme (inference'taki detect() ile birebir aynı olmalı)
# =============================================================================
def robust_normalize(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    std = X.std(axis=0)
    sd = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return np.clip((X - med) / sd, -50, 50)


def delta_t_feature(t):
    dt = np.diff(t, prepend=t[0])
    med = np.median(dt[1:]) if len(dt) > 1 else 1.0
    f = np.log1p(dt / max(med, 1e-12))
    f[0] = 0.0
    return f


# =============================================================================
# Tek bir eğitim örneği
# =============================================================================
def make_sample(rng, difficulty=0.5, n_channels=None, length=None, force_anomaly=False,
                domain=None, domain_scenarios_only=False):
    name = domain or str(rng.choice(DOMAIN_NAMES, p=DOMAIN_WEIGHTS))
    dom = DOMAINS[name]
    T = length or (MAX_T if rng.random() < 0.5 else int(rng.integers(MIN_T, MAX_T + 1)))
    k = n_channels or int(round(np.exp(rng.uniform(0, np.log(MAX_CH)))))   # az sütun daha sık
    step = float(rng.choice(dom["steps"]))
    has_anom = force_anomaly or rng.random() > CLEAN_RATIO

    # Veri boşluğu: fazladan satır üretip ortadan silinir (zaman akmaya devam eder)
    gap = int(rng.integers(10, 100)) if (has_anom and not domain_scenarios_only
                                         and rng.random() < MISSING_DATA_RATIO) else 0
    t = make_timestamps(rng, T + gap, step, dom.get("business_days", False), dom.get("jitter", True))
    extra = {}
    X, roles, ent = dom["gen"](rng, t, k, extra)
    keep = None
    if gap:
        gi = int(rng.integers(max(1, T // 10), T - 1))
        keep = np.r_[0:gi, gi + gap:T + gap]
        X, t = X[keep], t[keep]
    X = np.array(X, dtype=float)

    ctx = Ctx(rng, X, t, roles, ent, difficulty, extra, keep)
    if gap:
        ctx.mark(gi, gi + 1, list(range(k)), "missing_data")
    if has_anom:
        for _ in range(int(rng.integers(1, 4))):
            use_domain = dom["scenarios"] and (domain_scenarios_only or rng.random() < DOMAIN_SCENARIO_RATIO)
            operation = dom["scenarios"][int(rng.integers(len(dom["scenarios"])))] if use_domain else generic_anomaly
            if not safe_inject(ctx, operation) and not domain_scenarios_only:
                for _attempt in range(5):
                    if safe_inject(ctx, generic_anomaly):
                        break
    X, labels, types = ctx.X, ctx.labels, ctx.types

    # Sütun sırasını karıştır: model pozisyon ezberlemesin
    perm = rng.permutation(k)
    X, labels, types = X[:, perm], labels[:, perm], types[:, perm]
    roles = [roles[i] for i in perm]

    values = np.zeros((MAX_T, MAX_CH), dtype=np.float32)
    values[:T, :k] = robust_normalize(X)
    lab = np.zeros((MAX_T, MAX_CH), dtype=np.int8)
    lab[:T, :k] = labels
    typ = np.zeros((MAX_T, MAX_CH), dtype=np.int8)
    typ[:T, :k] = types
    dtf = np.zeros(MAX_T, dtype=np.float32)
    dtf[:T] = delta_t_feature(t)
    time_mask = np.zeros(MAX_T, dtype=bool)
    time_mask[:T] = True
    channel_mask = np.zeros(MAX_CH, dtype=bool)
    channel_mask[:k] = True

    return {
        "raw": np.column_stack([t, X]),   # (T, 1+k)  kullanıcının vereceği matris
        "values": values,                 # (2048, 100) normalize edilmiş değerler
        "delta_t": dtf,                   # (2048,)    zaman aralığı özelliği
        "time_mask": time_mask,           # (2048,)    hangi satırlar gerçek
        "channel_mask": channel_mask,     # (100,)     hangi sütunlar gerçek
        "labels": lab,                    # (2048, 100) 0/1 anomali
        "types": typ,                     # (2048, 100) anomali türü id
        "row_labels": lab.max(axis=1),    # (2048,)    satırda anomali var mı
        "meta": {"T": T, "k": k, "step_s": step, "domain": name, "roles": roles},
    }


# =============================================================================
# Örnek çalıştırma: istatistik, CSV ve alan panosu
# =============================================================================
if __name__ == "__main__":
    import time
    from collections import Counter
    import pandas as pd

    rng = np.random.default_rng(42)
    N = 3000
    t0 = time.time()
    dom_c, type_c, bad, clean, persist = Counter(), Counter(), 0, 0, 0
    for _ in range(N):
        s = make_sample(rng, difficulty=rng.uniform())
        m = s["meta"]
        dom_c[m["domain"]] += 1
        if not np.isfinite(s["values"]).all() or not np.isfinite(s["raw"]).all():
            bad += 1
        if s["labels"].sum() == 0:
            clean += 1
        elif s["labels"][m["T"] - 1].any():
            persist += 1
        for tid, cnt in zip(*np.unique(s["types"], return_counts=True)):
            if tid > 0:
                type_c[TYPES[tid]] += 1
    el = time.time() - t0
    print(f"{N} örnek {el:.1f} sn'de üretildi ({N / el:.0f} örnek/sn), bozuk değer içeren: {bad}")
    print(f"temiz örnek oranı: {clean / N:.0%}, anomalisi sona kadar süren: {persist / (N - clean):.0%}")
    print("alan dağılımı:", dict(dom_c))
    print("anomali türleri (kaç örnekte görüldü):", dict(type_c))

    # Kullanıcı biçiminde örnek CSV
    demo = make_sample(np.random.default_rng(7), difficulty=0.3, n_channels=4, length=300,
                       force_anomaly=True, domain="manufacturing", domain_scenarios_only=True)
    T, k = demo["meta"]["T"], demo["meta"]["k"]
    df = pd.DataFrame(demo["raw"][:, 1:], columns=[f"ch_{j}" for j in range(k)])
    df.insert(0, "timestamp", pd.to_datetime(demo["raw"][:, 0], unit="s"))
    ty = demo["types"][:T, :k]
    df["anomali"] = demo["row_labels"][:T]
    df["tur"] = [TYPES[r[r > 0][0]] if (r > 0).any() else "normal" for r in ty]
    df["sutunlar"] = [",".join(f"ch_{j}" for j in np.where(r > 0)[0]) or "-" for r in ty]
    df.to_csv("ornek_egitim_verisi.csv", index=False)
    print("\nornek_egitim_verisi.csv kaydedildi")

    # Her alandan bir örnek: alan panosu
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n in DOMAIN_NAMES if n != "abstract"]
        fig, axes = plt.subplots(len(names) // 2, 2, figsize=(16, 2.6 * (len(names) // 2)))
        colors = ["#2b6cb0", "#2f855a", "#b7791f", "#6b46c1"]
        for ax, name in zip(axes.ravel(), names):
            nch = 3 if name in ("ecg", "glucose") else 4
            for seed in range(50):
                d = make_sample(np.random.default_rng(1000 + seed), difficulty=0.2, n_channels=nch,
                                length=400, force_anomaly=True, domain=name, domain_scenarios_only=True)
                if d["labels"].sum() > 0:
                    break
            T_, k_ = d["meta"]["T"], d["meta"]["k"]
            x = d["values"][:T_, :k_]
            found = set()
            for j in range(k_):
                y = x[:, j] / (np.abs(x[:, j]).max() + 1e-9) * 0.45 - j
                ax.plot(y, lw=0.8, color=colors[j % 4])
                tj = d["types"][:T_, j]
                i = 0
                while i < T_:
                    if tj[i] > 0:
                        e2 = i
                        while e2 + 1 < T_ and tj[e2 + 1] == tj[i]:
                            e2 += 1
                        ax.axvspan(i, e2 + 1, ymin=0, ymax=1, color="#e53e3e", alpha=0.12)
                        ax.plot(range(i, e2 + 1), y[i:e2 + 1], lw=1.2, color="#c53030")
                        found.add(TYPES[tj[i]])
                        i = e2 + 1
                    else:
                        i += 1
            ax.set_yticks([-j for j in range(k_)])
            ax.set_yticklabels(d["meta"]["roles"], fontsize=7)
            ax.set_xticks([])
            ax.set_title(f"{name}  ({d['meta']['step_s']:g} sn aralık)  →  {', '.join(sorted(found))}", fontsize=9)
        fig.suptitle("Her alandan bir eğitim örneği (normalize, kırmızı: enjekte edilen anomali)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig("alan_ornekleri.png", dpi=110)
        print("alan_ornekleri.png kaydedildi")
    except ImportError:
        pass
