# Veri havuzu

Hugging Face kopyası (özel): `cagrigungor/tisan-havuz` — `data/havuz` ile birebir aynı içerik.
Toplam 12.470 seri, ~65M satır, 15 sektör. Kaynak başına satır: HAI 4.9M, LOTSA 57M, SMD 1.4M, SMAP/MSL 0.7M, wind 0.5M, NAB 0.4M, Pump 0.2M.

```
data/
├── raw/                 indirilen ham veri (dokunulmaz)
└── havuz/
    ├── katalog.csv      seri başına bilgi: kaynak, sektör, adım, etiket düzeyi, lisans, benchmark
    ├── gercek/          gerçek veri, kaynak başına bir Parquet (gercek_veri_hazirla.py üretir)
    └── sentetik/        sentetik veri (egitim_verisi_uretici.py çıktıları buraya)
```

## Format (uzun)

| Sütun | Tip | Anlam |
|---|---|---|
| `series_id` | str | `kaynak/.../bölüm`, ör. `smd/machine-1-1/test` |
| `timestamp` | float64 | unix saniye |
| `channel` | int16 | 0..k-1 (adlar katalogdaki `channel_names`) |
| `value` | float32 | ham değer, NaN olabilir |
| `label` | int8 | 1 anomali · 0 normal · -1 etiketsiz |

Katalogda `label_level`: `row` satır etiketi tüm sütunlara yayılmış demek, `cell` sütun bazında etiket demek.
`synthetic_time=True` ise kaynakta zaman damgası yoktu ve sabit adımla üretildi.

```python
from gercek_veri_hazirla import seri_yukle
s = seri_yukle("nab/realKnownCause/nyc_taxi")
s["raw"], s["labels"], s["meta"]
```

## Kaynaklar

| Kaynak | Seri | Sütun | Adım | Etiket | Lisans | Benchmark |
|---|---|---|---|---|---|---|
| SKAB | 35 | 8 | 1 sn | satır | AGPL-3.0 | SKAB |
| SKAB teaser | 1 | 8 | 1 sn | yok | AGPL-3.0 | – |
| NAB | 58 | 1 | değişken | satır (pencere) | AGPL-3.0 | TSB-AD-U |
| SMAP / MSL | 110 / 54 | 25 / 55 | sentetik | hücre (telemetri), train etiketsiz | telemanom | TSB-AD-M |
| Pump sensor | 1 | 51 | 1 dk | satır (BROKEN+RECOVERING) | belirsiz | – |
| SMD | 56 | 38 | sentetik 1 dk | hücre, train etiketsiz | MIT | TSB-AD-M |
| CNC mill | 18 | 47 | 0.1 sn | yok | CC0 | – |
| Wind gearbox SCADA | 2 | 7 | 10 dk | satır (labeled), yok (complex) | Apache-2.0 | – |
| HAI 20.07/21.03/22.04/23.05 | 28 | 60–86 | 1 sn | hücre (attack_P1–P3) / satır; train saldırısız | CC-BY-SA-4.0 | – |
| MetroPT-3 | 1 | 15 | 10 sn | satır (makaledeki 4 arıza) | CC-BY-4.0 | – |
| CATS | 2 (train/val) | 17 | 1 sn | hücre (kök neden kanalı) | CC-BY-4.0 | – |
| Tennessee Eastman | ~550 | 52 | 3 dk (sentetik zaman) | satır (20. örnekten sonra arıza) | CC-BY-4.0, simülasyon | – |
| LOTSA (100 alt küme, örneklenmiş) | 12.109 | 1–çok | değişken | yok | alt kümeye göre | – |

**Eğitim/doğrulama bölmesi (egitim.ipynb):** HAI 20.07 ve 21.03 test dosyaları gerçek etiketleriyle eğitimde; HAI 22.04/23.05 test,
wind `labeled`, Pump, MetroPT-3 ve CATS `val` doğrulamada. Amaç: modelin eğitimde gerçek anomali de görmesi, doğrulamanın farklı yıl/düzenekte kalması.

**Dikkat:** `benchmark` sütunu dolu olan seriler TSB-AD'de değerlendirme verisi. README §9 gereği
eğitime girerlerse o bölümler değerlendirmeden çıkarılmalı (veya tersi).
