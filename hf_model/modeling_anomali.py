"""
Zero-shot zaman serisi anomali modeli: ön işleme + iki eksenli dikkat + detect().

Uzak kod (trust_remote_code) olarak HF deposunda ağırlıklarla birlikte durur.
Sadece torch, numpy ve transformers kullanır. Ön işleme fonksiyonları eğitimde de
birebir aynı şekilde kullanılır (egitim.ipynb bunları buradan içe aktarır).
"""
import math

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
def robust_normalize(X):
    """Sütun bazında medyan/MAD normalizasyonu, ±50 kırpma. NaN'lar yok sayılır."""
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    std = np.nanstd(X, axis=0)
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


def prepare_window(t, X, max_t, max_ch):
    """(T, k) ham pencereyi model girdisine çevirir: normalize + sıfır dolgu + maskeler."""
    T, k = X.shape
    assert T <= max_t and k <= max_ch
    values = np.zeros((max_t, max_ch), dtype=np.float32)
    values[:T, :k] = robust_normalize(fill_nan(X))
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
    noisy_or: 1 - Π(1 - p): bağımsız kanıtları birleştirir"""
    P = np.asarray(cell_scores, dtype=np.float64)
    if P.ndim == 1 or P.shape[1] == 1:
        return P.reshape(len(P))
    if method == "max":
        return P.max(1)
    if method == "topk":
        kk = min(k, P.shape[1])
        return np.sort(P, axis=1)[:, -kk:].mean(1)
    if method == "noisy_or":
        return 1 - np.prod(1 - np.clip(P, 0, 1 - 1e-6), axis=1)
    raise ValueError(method)


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
class _Attention(nn.Module):
    """Çok başlı dikkat, F.scaled_dot_product_attention ile (flash / mem-efficient çekirdekler)."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dk = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, key_pad):
        # x: (N, L, d); key_pad: (N, L) True = dolgu
        N, L, d = x.shape
        q, k, v = self.qkv(x).reshape(N, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)   # 3 × (N, h, L, dk)
        mask = (~key_pad)[:, None, None, :]                                             # True = dikkat edilebilir
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

    def forward(self, h, patch_pad, ch_pad):
        # h: (B, P, C, d); patch_pad: (B, P) True = dolgu; ch_pad: (B, C) True = dolgu
        B, P, C, d = h.shape
        # zaman dikkati: her sütun kendi geçmişine bakar
        x = self.n_time(h).permute(0, 2, 1, 3).reshape(B * C, P, d)
        a = self.att_time(x, patch_pad.unsqueeze(1).expand(B, C, P).reshape(B * C, P))
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
        self.pos = nn.Parameter(torch.zeros(c.max_t // c.patch, c.d_model))
        self.blocks = nn.ModuleList([_Block(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)])
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.patch * (1 + c.n_types))
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
        if "diff" in self.config.extra_channels:
            d = torch.zeros_like(values)
            d[:, 1:] = values[:, 1:] - values[:, :-1]
            d = d * time_mask[:, :, None].to(values.dtype)
            feats.append(d)
        return torch.stack(feats, dim=-1)                       # (B, T, C, F)

    def forward(self, values, delta_t, time_mask, channel_mask, labels=None, types=None,
                focal_gamma=2.0, focal_alpha=0.75, type_weight=0.5):
        """
        values (B,T,C) float · delta_t (B,T) · time_mask (B,T) bool · channel_mask (B,C) bool
        T, patch'in katı olmalı; T ≤ max_t, C ≤ max_ch. Dolgu sağda ve sağ sütunlarda.
        labels (B,T,C) {0,1}; types (B,T,C) tür id, -1 = bilinmiyor (kayba girmez).
        """
        cfg = self.config
        B, T, C = values.shape
        p = cfg.patch
        P = T // p
        x = self._features(values, time_mask)                              # (B,T,C,F)
        x = x.reshape(B, P, p, C, self.n_feat).permute(0, 1, 3, 2, 4).reshape(B, P, C, p * self.n_feat)
        h = self.embed(x)
        h = h + self.dt_embed(delta_t.reshape(B, P, p))[:, :, None, :]
        h = h + self.pos[:P][None, :, None, :]
        patch_valid = time_mask.reshape(B, P, p).any(-1)
        patch_pad, ch_pad = ~patch_valid, ~channel_mask
        for blk in self.blocks:
            if self.training and getattr(self, "gradient_checkpointing", False):
                h = checkpoint(blk, h, patch_pad, ch_pad, use_reentrant=False)
            else:
                h = blk(h, patch_pad, ch_pad)
        out = self.head(self.norm(h))                                       # (B,P,C,p*(1+K))
        out = out.reshape(B, P, C, p, 1 + cfg.n_types).permute(0, 1, 3, 2, 4).reshape(B, T, C, 1 + cfg.n_types)
        logits, type_logits = out[..., 0], out[..., 1:]
        result = {"logits": logits, "type_logits": type_logits}
        if labels is not None:
            valid = time_mask[:, :, None] & channel_mask[:, None, :]
            y = labels.float()
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = torch.exp(-bce)
            alpha = torch.where(y > 0.5, torch.full_like(y, focal_alpha), torch.full_like(y, 1 - focal_alpha))
            focal = alpha * (1 - pt) ** focal_gamma * bce
            loss = (focal * valid).sum() / valid.sum().clamp(min=1)
            if types is not None:
                tmask = valid & (labels > 0) & (types > 0)
                if tmask.any():
                    ce = F.cross_entropy(type_logits[tmask], types[tmask].long(), reduction="mean")
                    loss = loss + type_weight * ce
            result["loss"] = loss
        return result

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def score_matrix(self, t, X, batch_size=8, stride=None):
        """(T, k) ham matris → (T, k) kalibre hücre olasılığı ve (T, k) tür id.
        Uzun veri kayan pencere, çok sütun 100'lük gruplarla işlenir."""
        cfg = self.config
        T, k = X.shape
        stride = stride or cfg.max_t // 2
        dev = next(self.parameters()).device
        prob = np.zeros((T, k), dtype=np.float32)
        tsum = np.zeros((T, k, cfg.n_types), dtype=np.float32)
        wsum = np.zeros((T, 1), dtype=np.float32)
        starts = [0] if T <= cfg.max_t else list(range(0, T - cfg.max_t + 1, stride))
        if T > cfg.max_t and starts[-1] + cfg.max_t < T:
            starts.append(T - cfg.max_t)
        ch_groups = [list(range(a, min(a + cfg.max_ch, k))) for a in range(0, k, cfg.max_ch)]
        jobs = [(s, g) for s in starts for g in ch_groups]
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            vals, dts, tms, cms = zip(*[prepare_window(t[s:s + cfg.max_t], X[s:s + cfg.max_t][:, g], cfg.max_t, cfg.max_ch)
                                        for s, g in chunk])
            with torch.autocast(dev.type if hasattr(dev, "type") else str(dev).split(":")[0], dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                out = self(torch.tensor(np.stack(vals), device=dev), torch.tensor(np.stack(dts), device=dev),
                           torch.tensor(np.stack(tms), device=dev), torch.tensor(np.stack(cms), device=dev))
            temp = cfg.temperature if np.isfinite(cfg.temperature) and cfg.temperature > 0 else 1.0
            pr = torch.sigmoid(out["logits"] / temp).float().cpu().numpy()
            tp = torch.softmax(out["type_logits"], -1).float().cpu().numpy()
            for (s, g), pw, tw in zip(chunk, pr, tp):
                L = min(cfg.max_t, T - s)
                # pencere ortası daha güvenilir: üçgen ağırlık
                w = (1 - np.abs(np.linspace(-1, 1, L))) * 0.9 + 0.1
                prob[s:s + L, g] += pw[:L, :len(g)] * w[:, None]
                tsum[s:s + L, g] += tw[:L, :len(g)] * w[:, None, None]
                wsum[s:s + L] += w[:, None] / len(ch_groups)
        prob /= np.maximum(wsum, 1e-8)
        return prob, tsum.argmax(-1)

    def detect(self, matrix, sensitivity="medium", batch_size=8):
        """Kullanıcı arayüzü. matrix: (T, 1+k); ilk sütun zaman damgası."""
        cfg = self.config
        M = np.asarray(matrix)
        if M.ndim != 2 or M.shape[1] < 2:
            raise ValueError("matris (T, 1+k) olmalı: ilk sütun zaman, en az bir değer sütunu")
        t = parse_time_column(M[:, 0])
        X = M[:, 1:].astype(np.float64)
        order = np.argsort(t, kind="stable")
        t, X = t[order], X[order]
        # tekrarlı zaman damgaları: ortalama
        uniq, inv = np.unique(t, return_inverse=True)
        if len(uniq) < len(t):
            Xa = np.zeros((len(uniq), X.shape[1]))
            cnt = np.zeros(len(uniq))
            np.add.at(Xa, inv, np.nan_to_num(X))
            np.add.at(cnt, inv, 1)
            X, t = Xa / cnt[:, None], uniq
        T, k = X.shape
        thr = {"low": 0.9, "medium": 0.7, "high": 0.5}[sensitivity]
        if T < cfg.min_t:
            # istatistiksel yedek: robust z-skoru
            z = np.abs(robust_normalize(fill_nan(X)))
            prob = np.clip(1 - np.exp(-np.maximum(z - 3, 0)), 0, 1).astype(np.float32)
            types = np.where(prob > thr, cfg.type_names.index("spike"), 0)
            note = f"{T} satır < {cfg.min_t}: model yerine robust z-skoru kullanıldı"
        else:
            prob, types = self.score_matrix(t, X, batch_size=batch_size)
            note = None
        return AnomaliResult(t, X, order, prob, types, thr, cfg.type_names, note,
                             row_agg=cfg.row_agg, row_topk=cfg.row_topk)


class AnomaliResult:
    def __init__(self, t, X, order, cell_scores, cell_types, threshold, type_names, note=None,
                 row_agg="topk", row_topk=3):
        self.timestamps, self.values, self.order = t, X, order
        self.cell_scores, self.cell_types = cell_scores, cell_types
        self.threshold, self.type_names, self.note = threshold, type_names, note
        self.row_scores = aggregate_rows(cell_scores, row_agg, row_topk)
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
            cells = self.cell_scores[s:e + 1] > self.threshold
            chans = np.where(cells.any(0))[0].tolist()
            ty = self.cell_types[s:e + 1][cells]
            kind = self.type_names[int(np.bincount(ty).argmax())] if len(ty) else "unknown"
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
        for c in range(k):
            axes[c].plot(x, self.values[:, c], lw=0.8)
            m = self.cell_scores[:, c] > self.threshold
            axes[c].scatter(x[m], self.values[m, c], color="red", s=8)
            axes[c].set_ylabel(f"ch_{c}")
        axes[-1].plot(x, self.row_scores, color="black", lw=0.8)
        axes[-1].axhline(self.threshold, color="red", ls="--", lw=0.8)
        axes[-1].set_ylabel("score")
        plt.tight_layout()
        return fig

    def __repr__(self):
        return f"AnomaliResult(rows={len(self.row_scores)}, anomaly_rows={len(self.anomaly_rows)}, events={len(self.events)})"
