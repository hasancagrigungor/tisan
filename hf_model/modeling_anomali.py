"""
Zero-shot zaman serisi anomali modeli: ön işleme + iki eksenli dikkat + detect().

Uzak kod (trust_remote_code) olarak HF deposunda ağırlıklarla birlikte durur.
Sadece torch, numpy ve transformers kullanır. Ön işleme fonksiyonları eğitimde de
birebir aynı şekilde kullanılır (egitim.ipynb bunları buradan içe aktarır).
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .configuration_anomali import AnomaliConfig


# =============================================================================
# Ön işleme (eğitim ve inference için ortak, NumPy)
# =============================================================================
def robust_normalize(X, reference=None):
    """Sütun bazında medyan/MAD normalizasyonu, ±50 kırpma. NaN'lar yok sayılır."""
    X = np.asarray(X, dtype=np.float64)
    # GPT-6 Astra: isteğe bağlı doğrulanmış normal referans kalıcı seviye farkını korur.
    base = X if reference is None else np.asarray(reference, dtype=np.float64)
    med = np.nanmedian(base, axis=0)
    mad = np.nanmedian(np.abs(base - med), axis=0) * 1.4826
    std = np.nanstd(base, axis=0)
    sd = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return np.clip((X - med) / sd, -50, 50)


def delta_t_feature(t):
    """log1p(Δt / medyan Δt); ilk satır 0. Boşluklar büyük, düzenli akış ~log(2) verir."""
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    med = np.median(dt[1:]) if len(dt) > 1 else 1.0
    f = np.log1p(dt / max(med, 1e-12))
    f[0] = 0.0
    return f


def fill_nan(X):
    """İleri doldurma, sonra sütun medyanı; tamamen boş sütun 0 olur."""
    X = np.array(X, dtype=np.float64, copy=True)
    for c in range(X.shape[1]):
        col = X[:, c]
        m = np.isnan(col)
        if m.all():
            col[:] = 0.0
            continue
        if m.any():
            idx = np.where(~m, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]
            col[np.isnan(col)] = np.nanmedian(col)
    return X


def prepare_window(t, X, max_t, max_ch, reference=None):
    """(T, k) ham pencereyi model girdisine çevirir: normalize + sıfır dolgu + maskeler."""
    T, k = X.shape
    assert T <= max_t and k <= max_ch
    values = np.zeros((max_t, max_ch), dtype=np.float32)
    values[:T, :k] = robust_normalize(fill_nan(X), reference=reference)
    dtf = np.zeros(max_t, dtype=np.float32)
    dtf[:T] = delta_t_feature(t)
    time_mask = np.zeros(max_t, dtype=bool)
    time_mask[:T] = True
    channel_mask = np.zeros(max_ch, dtype=bool)
    channel_mask[:k] = True
    return values, dtf, time_mask, channel_mask


def aggregate_rows(cell_scores, method="topk", k=3):
    """(T, k) hücre olasılığı → (T,) satır skoru.
    max: en yüksek hücre (çok sütunda yanlış pozitife açık)
    topk: en yüksek k hücrenin ortalaması (k > sütun sayısıysa max'a düşer)
    noisy_or: en yüksek k hücrede 1 - Π(1 - p); tüm kanallarda alınırsa 20+ kanalda 1'e doyar"""
    P = np.asarray(cell_scores, dtype=np.float64)
    if P.ndim == 1 or P.shape[1] == 1:
        return P.reshape(len(P))
    if method == "max":
        return P.max(1)
    if method == "topk":
        kk = min(k, P.shape[1])
        return np.sort(P, axis=1)[:, -kk:].mean(1)
    if method == "noisy_or":                                            # en yüksek k hücre üzerinden: çok kanalda doymaz
        kk = min(max(k, 1), P.shape[1])
        top = np.sort(np.clip(P, 0, 1 - 1e-6), axis=1)[:, -kk:]
        return 1 - np.prod(1 - top, axis=1)
    raise ValueError(method)


def calibrate_rows(scores, temperature=1.0, bias=0.0):
    """GPT-6 Astra: toplulaştırma SONRASI, ayrı gerçek veride öğrenilen satır kalibrasyonu."""
    if temperature == 1.0 and bias == 0.0:
        return np.asarray(scores)
    p = np.clip(scores, 1e-7, 1 - 1e-7)
    z = (np.log(p) - np.log1p(-p)) / temperature + bias
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def parse_time_column(col):
    """Unix saniye / milisaniye / ISO metin → unix saniye (float)."""
    col = np.asarray(col)
    if np.issubdtype(col.dtype, np.datetime64):
        return col.astype("datetime64[ns]").astype(np.int64) / 1e9
    if col.dtype.kind in "iuf":
        t = col.astype(np.float64)
        if np.nanmedian(np.abs(t)) > 1e11:      # milisaniye
            t = t / 1000.0
        return t
    return np.array(col.astype(str), dtype="datetime64[ns]").astype(np.int64) / 1e9


# =============================================================================
# Model
# =============================================================================
def _rope(q, k, pos, base=10000.0):
    """Döner konum kodlaması (RoPE), sürekli konumlarla. q,k: (N, h, L, dk); pos: (N, L) float."""
    dk = q.shape[-1]
    half = dk // 2
    freq = base ** (-torch.arange(half, device=q.device, dtype=torch.float32) / half)       # (half,)
    ang = pos.float()[:, None, :, None] * freq[None, None, None, :]                          # (N,1,L,half)
    cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:2 * half]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half:]], dim=-1)
    return rot(q), rot(k)


class _Attention(nn.Module):
    """Çok başlı dikkat, F.scaled_dot_product_attention ile (flash / mem-efficient çekirdekler)."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, key_pad, pos=None):
        # x: (N, L, d); key_pad: (N, L) True = dolgu; pos: (N, L) RoPE için sürekli konum (None = konumsuz)
        N, L, d = x.shape
        q, k, v = self.qkv(x).reshape(N, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # 3 × (N, h, L, dk)
        if pos is not None:
            q, k = _rope(q, k, pos)
        # dolgu yoksa maske verilmez: SDPA flash çekirdeğini kullanabilir (maske varken mem-efficient/matematik çekirdeğe düşer)
        mask = None if not bool(key_pad.any()) else (~key_pad)[:, None, None, :]         # True = dikkat edilebilir
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(a.transpose(1, 2).reshape(N, L, d))


class _Block(nn.Module):
    """Pre-norm: [zaman dikkati → sütun dikkati → FFN]."""

    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.n_time = nn.LayerNorm(d)
        self.att_time = _Attention(d, n_heads, dropout)
        self.n_ch = nn.LayerNorm(d)
        self.att_ch = _Attention(d, n_heads, dropout)
        self.n_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, patch_pad, ch_pad, pos=None):
        # h: (B, P, C, d); patch_pad: (B, P) True = dolgu; ch_pad: (B, C) True = dolgu; pos: (B, P) zaman konumu
        B, P, C, d = h.shape
        # zaman dikkati: her sütun kendi geçmişine bakar
        x = self.n_time(h).permute(0, 2, 1, 3).reshape(B * C, P, d)
        a = self.att_time(x, patch_pad.unsqueeze(1).expand(B, C, P).reshape(B * C, P),
                          None if pos is None else pos.unsqueeze(1).expand(B, C, P).reshape(B * C, P))
        h = h + self.drop(a.reshape(B, C, P, d).permute(0, 2, 1, 3))
        # sütun dikkati: aynı andaki sütunlar birbirine bakar (pozisyon bilgisi yok)
        x = self.n_ch(h).reshape(B * P, C, d)
        a = self.att_ch(x, ch_pad.unsqueeze(1).expand(B, P, C).reshape(B * P, C))
        h = h + self.drop(a.reshape(B, P, C, d))
        h = h + self.drop(self.ff(self.n_ff(h)))
        return h


class AnomaliModel(PreTrainedModel):
    config_class = AnomaliConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        c = config
        self.n_feat = 1 + len(c.extra_channels)
        self.embed = nn.Linear(c.patch * self.n_feat, c.d_model)
        self.dt_embed = nn.Linear(c.patch, c.d_model)
        self.pos = nn.Parameter(torch.zeros(c.max_t // c.patch, c.d_model))     # "learned" modunda kullanılır
        self.mask_token = nn.Parameter(torch.zeros(c.d_model))                   # maskeli yeniden inşa için
        self.blocks = nn.ModuleList([_Block(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)])
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.patch * (1 + c.n_types))
        self.recon = nn.Linear(c.d_model, c.patch) if getattr(c, "recon_head", False) else None
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # --- girdi özellikleri ---
    def _features(self, values, time_mask):
        feats = [values]
        m = time_mask[:, :, None].to(values.dtype)
        for name in self.config.extra_channels:
            if name == "diff":
                d = torch.zeros_like(values)
                d[:, 1:] = values[:, 1:] - values[:, :-1]
                feats.append(d * m)
            elif name.startswith("ms"):                          # msN: N satırlık ortalanmış, maske ağırlıklı kayan ortalama
                n = int(name[2:])
                B, T, C = values.shape
                v = (values * m).permute(0, 2, 1).reshape(B * C, 1, T)
                w = m.expand(B, T, C).permute(0, 2, 1).reshape(B * C, 1, T)
                num = F.avg_pool1d(v, n, stride=1, padding=n // 2, count_include_pad=True)[:, :, :T]
                den = F.avg_pool1d(w, n, stride=1, padding=n // 2, count_include_pad=True)[:, :, :T]
                ms = (num / den.clamp(min=1e-6)).reshape(B, C, T).permute(0, 2, 1)
                feats.append((values - ms) * m)                  # yerel seviyeden sapma: uzun bağlamda kayma görünür
        return torch.stack(feats, dim=-1)                       # (B, T, C, F)

    def forward(self, values, delta_t, time_mask, channel_mask, labels=None, types=None,
                focal_gamma=2.0, focal_alpha=0.75, type_weight=0.5,
                label_weights=None, row_labels=None, row_weights=None, mask_patches=None, recon_weight=1.0):
        """
        values (B,T,C) float · delta_t (B,T) · time_mask (B,T) bool · channel_mask (B,C) bool
        T, patch'in katı olmalı; T ≤ max_t, C ≤ max_ch. Dolgu sağda ve sağ sütunlarda.
        labels (B,T,C) {0,1}; types (B,T,C) tür id, -1 = bilinmiyor (kayba girmez).
        """
        cfg = self.config
        B, T, C = values.shape
        p = cfg.patch
        P = T // p
        feat_in = values
        if mask_patches is not None:                                        # maskeli satırlar özelliklere sızmasın (diff/msN komşu patch'lere taşır)
            feat_in = values.masked_fill(mask_patches.repeat_interleave(p, dim=1), 0.0)
        x = self._features(feat_in, time_mask)                             # (B,T,C,F)
        x = x.reshape(B, P, p, C, self.n_feat).permute(0, 1, 3, 2, 4).reshape(B, P, C, p * self.n_feat)
        h = self.embed(x)
        if mask_patches is not None:                                        # maskeli yeniden inşa: girdi yerine mask token
            h = torch.where(mask_patches[..., None], self.mask_token.to(h.dtype).expand_as(h), h)
        h = h + self.dt_embed(delta_t.reshape(B, P, p))[:, :, None, :]
        pos = None
        if getattr(cfg, "pos_encoding", "learned") == "rope":
            # gerçek zaman konumu: Δt oranlarının kümülatif toplamı (medyan adım = 1); patch başına ortalama
            step = torch.expm1(delta_t.float()).clamp(min=0)
            step[:, 0] = 0
            pos = torch.cumsum(step, dim=1).reshape(B, P, p).mean(-1)      # (B, P)
        else:
            h = h + self.pos[:P][None, :, None, :]
        patch_valid = time_mask.reshape(B, P, p).any(-1)
        patch_pad, ch_pad = ~patch_valid, ~channel_mask
        for blk in self.blocks:
            if self.training and getattr(self, "gradient_checkpointing", False):
                h = checkpoint(blk, h, patch_pad, ch_pad, pos, use_reentrant=False)
            else:
                h = blk(h, patch_pad, ch_pad, pos)
        h = self.norm(h)
        recon = None
        if self.recon is not None:
            recon = self.recon(h).reshape(B, P, C, p).permute(0, 1, 3, 2).reshape(B, T, C)   # normalize değer tahmini
        out = self.head(h)                                                  # (B,P,C,p*(1+K))
        out = out.reshape(B, P, C, p, 1 + cfg.n_types).permute(0, 1, 3, 2, 4).reshape(B, T, C, 1 + cfg.n_types)
        logits, type_logits = out[..., 0], out[..., 1:]
        result = {"logits": logits, "type_logits": type_logits, "recon": recon}
        if mask_patches is not None and recon is not None:                  # ön eğitim / yardımcı kayıp
            valid = time_mask[:, :, None] & channel_mask[:, None, :]
            mrow = mask_patches.repeat_interleave(p, dim=1) & valid          # (B,T,C) maskeli & geçerli hücreler
            if mrow.any():
                result["recon_loss"] = F.smooth_l1_loss(recon[mrow], values[mrow].to(recon.dtype))
                result["loss"] = recon_weight * result["recon_loss"]
        if labels is not None:
            valid = time_mask[:, :, None] & channel_mask[:, None, :]
            # GPT-6 Astra: bilinmeyen ve satır düzeyindeki etiketler hücre hedefi değildir.
            y = labels.clamp(min=0).float()
            weights = valid * (labels >= 0)
            if label_weights is not None:
                weights = weights * label_weights
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = torch.exp(-bce)
            alpha = torch.where(y > 0.5, focal_alpha, 1 - focal_alpha)
            focal = alpha * (1 - pt) ** focal_gamma * bce
            # Güven ağırlığını paydada iptal etme: zayıf örnek gerçekten daha az katkı verir.
            loss = (focal * weights).sum() / valid.sum().clamp(min=1)
            if row_labels is not None and row_weights is not None:
                # En az bir kanal anormal: satır etiketi tüm hücreleri pozitif yapmaz.
                row_logits = logits.masked_fill(~channel_mask[:, None, :], -1e4).amax(-1)
                rbce = F.binary_cross_entropy_with_logits(row_logits, row_labels.clamp(min=0).float(), reduction="none")
                rw = row_weights * time_mask * (row_labels >= 0)
                loss = loss + (rbce * rw).sum() / time_mask.sum().clamp(min=1)
            if types is not None and getattr(cfg, "use_types", False) and type_weight > 0:
                tmask = valid & (labels > 0) & (types > 0) & (weights > 0)
                if tmask.any():
                    ce = F.cross_entropy(type_logits[tmask], types[tmask].long(), reduction="mean")
                    loss = loss + type_weight * ce
            if "loss" in result:                                            # ön eğitim kaybı + denetimli kayıp
                loss = loss + result["loss"]
            result["loss"] = loss
        return result

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def _score_windows(self, jobs, t, X, batch_size, ref_of):
        """jobs: [(s, g)]; ref_of(s, g) → (T_ref, |g|) ham referans ya da None."""
        cfg = self.config
        dev = next(self.parameters()).device
        out_pr, out_tp = [], []
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            vals, dts, tms, cms = zip(*[prepare_window(t[s:s + cfg.max_t], X[s:s + cfg.max_t][:, g], cfg.max_t, cfg.max_ch,
                                                       reference=ref_of(s, g)) for s, g in chunk])
            with torch.autocast(dev.type if hasattr(dev, "type") else str(dev).split(":")[0], dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                out = self(torch.tensor(np.stack(vals), device=dev), torch.tensor(np.stack(dts), device=dev),
                           torch.tensor(np.stack(tms), device=dev), torch.tensor(np.stack(cms), device=dev))
            temp = cfg.temperature if np.isfinite(cfg.temperature) and cfg.temperature > 0 else 1.0
            out_pr += list(torch.sigmoid(out["logits"] / temp).float().cpu().numpy())
            out_tp += list(torch.softmax(out["type_logits"], -1).float().cpu().numpy())
        return out_pr, out_tp

    @torch.no_grad()
    def score_matrix(self, t, X, batch_size=8, stride=None, reference=None, multi_scale=None):
        """(T, k) ham matris → (T, k) hücre olasılığı ve (T, k) tür id. Pencereden uzun seride config.multi_scale
        (veya multi_scale) katsayılarıyla seyreltilmiş seri de skorlanır; olasılıklar multi_scale_agg ile birleşir."""
        cfg = self.config
        prob, types = self._score_single(t, X, batch_size, stride, reference)
        factors = cfg.multi_scale if multi_scale is None else multi_scale
        T, k = X.shape
        for f in factors or ():
            f = int(f)
            n = T // f
            if T <= cfg.max_t or f < 2 or n < cfg.min_t:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)       # tümü NaN blok → NaN (fill_nan doldurur)
                Xd = np.nanmean(X[:n * f].reshape(n, f, k), axis=1)
            pd_, _ = self._score_single(np.asarray(t, dtype=np.float64)[:n * f:f], Xd, batch_size, None, None)
            up = np.repeat(pd_, f, axis=0)
            up = np.concatenate([up, np.repeat(up[-1:], T - len(up), axis=0)]) if len(up) < T else up
            prob = np.maximum(prob, up) if cfg.multi_scale_agg == "max" else (prob + up) / 2
        return prob, types

    def _score_single(self, t, X, batch_size=8, stride=None, reference=None):
        """Tek çözünürlük. Uzun veri kayan pencere,
        çok sütun 100'lük gruplarla işlenir.
        Referans normalizasyonu: `reference` verilmişse her pencere onunla normalize edilir. Verilmemiş ve
        config.auto_reference açıksa pencereler zaman sırasıyla işlenir; referans, son "normal" görünen
        pencerenin ham değerleridir ve anomali sürdüğü sürece dondurulur (kalıcı anomali "yeni normal" olmaz)."""
        cfg = self.config
        T, k = X.shape
        if reference is not None:
            reference = np.asarray(reference, dtype=np.float64)
            if reference.ndim != 2 or reference.shape[1] != k or len(reference) < cfg.min_t or not np.isfinite(reference).all():
                raise ValueError("normal referans en az min_t satır, aynı sütunlar ve sonlu değerler içermeli")
        stride = stride or cfg.max_t // 2
        prob = np.zeros((T, k), dtype=np.float32)
        tsum = np.zeros((T, k, cfg.n_types), dtype=np.float32)
        wsum = np.zeros((T, 1), dtype=np.float32)
        starts = [0] if T <= cfg.max_t else list(range(0, T - cfg.max_t + 1, stride))
        if T > cfg.max_t and starts[-1] + cfg.max_t < T:
            starts.append(T - cfg.max_t)
        ch_groups = [list(range(a, min(a + cfg.max_ch, k))) for a in range(0, k, cfg.max_ch)]

        def accumulate(s, g, pw, tw):
            L = min(cfg.max_t, T - s)
            w = (1 - np.abs(np.linspace(-1, 1, L))) * 0.9 + 0.1          # pencere ortası daha güvenilir
            prob[s:s + L, g] += pw[:L, :len(g)] * w[:, None]
            tsum[s:s + L, g] += tw[:L, :len(g)] * w[:, None, None]
            wsum[s:s + L] += w[:, None] / len(ch_groups)

        auto = reference is None and cfg.auto_reference and len(starts) > 1
        if not auto:
            jobs = [(s, g) for s in starts for g in ch_groups]
            ref_of = (lambda s, g: None) if reference is None else (lambda s, g: reference[:, g])
            prs, tps = self._score_windows(jobs, t, X, batch_size, ref_of)
            for (s, g), pw, tw in zip(jobs, prs, tps):
                accumulate(s, g, pw, tw)
        else:
            # zaman sırasında, grup başına dondurulabilir referans; ardışık pencereler batch'lenmez (bağımlılık var)
            for g in ch_groups:
                ref = None
                for s in starts:
                    pw, tw = self._score_windows([(s, g)], t, X, 1, lambda s_, g_: ref)
                    pw, tw = pw[0], tw[0]
                    accumulate(s, g, pw, tw)
                    L = min(cfg.max_t, T - s)
                    tail = aggregate_rows(pw[L * 3 // 4:L, :len(g)], cfg.row_agg, cfg.row_topk)
                    if tail.max() < cfg.reference_threshold:              # pencere sonu normal → referansı ilerlet
                        ref = fill_nan(X[s:s + L][:, g])
                    # aksi hâlde referans dondurulur: olay sürerken "yeni normal" öğrenilmez
        prob /= np.maximum(wsum, 1e-8)
        return prob, tsum.argmax(-1)

    def predict(self, matrix, sensitivity="medium", batch_size=8, normal_reference=None):
        """Birincil API: (T, 1+k) matris → (T,) 0/1 vektörü. 1 = bu satırda anomali var.
        Alan bilgisi gerekmez; finans, sensör, siber güvenlik ya da bilinmeyen veri aynı şekilde işlenir."""
        r = self.detect(matrix, sensitivity=sensitivity, batch_size=batch_size, normal_reference=normal_reference)
        out = np.zeros(len(r.row_scores), dtype=np.int8)
        out[r.anomaly_rows] = 1
        return out[r.row_index]                                              # girdiyle aynı uzunluk ve sıra

    def detect(self, matrix, sensitivity="medium", batch_size=8, normal_reference=None):
        """Kullanıcı arayüzü. matrix: (T, 1+k); ilk sütun zaman damgası."""
        cfg = self.config
        M = np.asarray(matrix)
        if M.ndim != 2 or M.shape[1] < 2:
            raise ValueError("matris (T, 1+k) olmalı: ilk sütun zaman, en az bir değer sütunu")
        t = parse_time_column(M[:, 0])
        X = M[:, 1:].astype(np.float64)
        order = np.argsort(t, kind="stable")
        t, X = t[order], X[order]
        # tekrarlı zaman damgaları: ortalama; row_index: kullanıcının i. satırı → sonuç satırı
        uniq, inv = np.unique(t, return_inverse=True)
        row_index = np.empty(len(order), dtype=np.int64)
        row_index[order] = inv
        if len(uniq) < len(t):
            Xa = np.zeros((len(uniq), X.shape[1]))
            cnt = np.zeros(len(uniq))
            np.add.at(Xa, inv, np.nan_to_num(X))
            np.add.at(cnt, inv, 1)
            X, t = Xa / cnt[:, None], uniq
        T, k = X.shape
        base = float(getattr(cfg, "row_threshold", 0.5))
        thr = {"low": base + (1 - base) * 0.5, "medium": base, "high": base * 0.6}[sensitivity]
        if T < cfg.min_t:
            # istatistiksel yedek: robust z-skoru
            z = np.abs(robust_normalize(fill_nan(X)))
            prob = np.clip(1 - np.exp(-np.maximum(z - 3, 0)), 0, 1).astype(np.float32)
            types = np.where(prob > thr, cfg.type_names.index("spike"), 0)
            note = f"{T} satır < {cfg.min_t}: model yerine robust z-skoru kullanıldı"
        else:
            prob, types = self.score_matrix(t, X, batch_size=batch_size, reference=normal_reference)
            note = None
        if not getattr(cfg, "use_types", False):
            types = np.zeros_like(types)                                        # tür başlığı kapalı: tür bilgisi yok
        return AnomaliResult(t, X, row_index, prob, types, thr, cfg.type_names, note,
                             row_agg=cfg.row_agg, row_topk=cfg.row_topk,
                             row_temperature=cfg.row_temperature if T >= cfg.min_t else 1.0,
                             row_bias=cfg.row_bias if T >= cfg.min_t else 0.0)


class AnomaliResult:
    def __init__(self, t, X, row_index, cell_scores, cell_types, threshold, type_names, note=None,
                 row_agg="max", row_topk=3, row_temperature=1.0, row_bias=0.0):
        self.timestamps, self.values = t, X
        self.row_index = row_index            # kullanıcının girdi satırı → sonuç satırı (sıralama + tekrar birleştirme)
        self.cell_scores, self.cell_types = cell_scores, cell_types
        self.threshold, self.type_names, self.note = threshold, type_names, note
        self.row_scores = calibrate_rows(aggregate_rows(cell_scores, row_agg, row_topk), row_temperature, row_bias)
        # eşik kalibre ölçekte: hücreler de aynı dönüşümle karşılaştırılır (ham olasılık ~0.5 > eşik → her hücre "anomali" hatası)
        self.cell_calibrated = calibrate_rows(cell_scores, row_temperature, row_bias)
        self.anomaly_rows = np.where(self.row_scores > threshold)[0].tolist()
        self.events = self._events()

    def _events(self):
        rows = np.array(self.anomaly_rows)
        events = []
        if len(rows) == 0:
            return events
        breaks = np.where(np.diff(rows) > 1)[0]
        for seg in np.split(rows, breaks + 1):
            s, e = int(seg[0]), int(seg[-1])
            cells = self.cell_calibrated[s:e + 1] > self.threshold
            if not cells.any():                                             # satır skoru top-k: tek hücre eşiği aşmayabilir
                cells = self.cell_calibrated[s:e + 1] >= self.cell_calibrated[s:e + 1].max(1, keepdims=True)
            chans = np.where(cells.any(0))[0].tolist()
            ty = self.cell_types[s:e + 1][cells]
            kind = self.type_names[int(np.bincount(ty).argmax())] if len(ty) and ty.max() > 0 else "anomaly"
            events.append({"start": s, "end": e, "start_time": float(self.timestamps[s]), "end_time": float(self.timestamps[e]),
                           "type": kind, "channels": chans, "confidence": float(self.row_scores[s:e + 1].max()),
                           "reason": f"{len(chans)} sütunda {kind}; {e - s + 1} satır boyunca beklenen davranıştan sapma"})
        return events

    def to_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.values, columns=[f"ch_{i}" for i in range(self.values.shape[1])])
        df.insert(0, "timestamp", pd.to_datetime(self.timestamps, unit="s"))
        df["score"] = self.row_scores
        df["anomaly"] = self.row_scores > self.threshold
        return df

    def plot(self, max_channels=8):
        import matplotlib.pyplot as plt
        k = min(self.values.shape[1], max_channels)
        fig, axes = plt.subplots(k + 1, 1, figsize=(12, 1.6 * (k + 1)), sharex=True)
        x = np.arange(len(self.timestamps))
        row_on = np.zeros(len(x), dtype=bool)
        row_on[self.anomaly_rows] = True
        for c in range(k):
            axes[c].plot(x, self.values[:, c], lw=0.8)
            m = row_on & (self.cell_calibrated[:, c] > self.threshold)     # yalnızca anomali satırlarında eşiği aşan hücre
            axes[c].scatter(x[m], self.values[m, c], color="red", s=8)
            axes[c].set_ylabel(f"ch_{c}")
        for ax in axes:
            for ev in self.events:
                ax.axvspan(ev["start"], ev["end"] + 1, color="red", alpha=0.08, lw=0)
        axes[-1].plot(x, self.row_scores, color="black", lw=0.8)
        axes[-1].axhline(self.threshold, color="red", ls="--", lw=0.8)
        axes[-1].set_ylabel("score")
        plt.tight_layout()
        return fig

    def __repr__(self):
        return f"AnomaliResult(rows={len(self.row_scores)}, anomaly_rows={len(self.anomaly_rows)}, events={len(self.events)})"
