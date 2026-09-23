from transformers import PretrainedConfig

TYPES = ["normal", "spike", "level_shift", "flatline", "drift", "noise_burst",
         "pattern_change", "correlation_break", "missing_data"]


class AnomaliConfig(PretrainedConfig):
    """Zero-shot zaman serisi anomali modeli ayarları."""
    model_type = "anomali"

    def __init__(self, d_model=256, n_layers=6, n_heads=8, d_ff=None, dropout=0.1,
                 patch=16, max_t=2048, max_ch=100, min_t=20,
                 extra_channels=("diff",), type_names=tuple(TYPES),
                 temperature=1.0, row_agg="topk", row_topk=3,
                 row_temperature=1.0, row_bias=0.0, auto_reference=True, reference_threshold=0.5, **kwargs):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.patch = patch
        self.max_t = max_t
        self.max_ch = max_ch
        self.min_t = min_t
        self.extra_channels = list(extra_channels)   # "diff": birinci fark kanalı
        self.type_names = list(type_names)
        self.temperature = temperature                # kalibrasyon (temperature scaling)
        self.row_agg = row_agg                        # hücre → satır skoru: "max" | "topk" | "noisy_or"
        self.row_topk = row_topk
        # GPT-6 Astra: hücre ve satır kalibrasyonu ayrı; eski checkpoint için özdeş dönüşüm.
        self.row_temperature = row_temperature
        self.row_bias = row_bias
        # Kalıcı anomali: kayan pencerede normalizasyon referansı, son "normal" görünen pencereden alınır
        # ve anomali sürerken dondurulur (olay sürekliliği). Kullanıcı normal_reference verirse o kullanılır.
        self.auto_reference = auto_reference
        self.reference_threshold = reference_threshold
        super().__init__(**kwargs)

    @property
    def n_types(self):
        return len(self.type_names)
