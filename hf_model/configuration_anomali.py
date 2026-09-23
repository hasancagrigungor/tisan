from transformers import PretrainedConfig

TYPES = ["normal", "spike", "level_shift", "flatline", "drift", "noise_burst",
         "pattern_change", "correlation_break", "missing_data"]


class AnomaliConfig(PretrainedConfig):
    """Zero-shot zaman serisi anomali modeli ayarları."""
    model_type = "anomali"

    def __init__(self, d_model=256, n_layers=6, n_heads=8, d_ff=None, dropout=0.1,
                 patch=16, max_t=2048, max_ch=100, min_t=20,
                 extra_channels=("diff",), type_names=tuple(TYPES),
                 temperature=1.0, **kwargs):
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
        super().__init__(**kwargs)

    @property
    def n_types(self):
        return len(self.type_names)
