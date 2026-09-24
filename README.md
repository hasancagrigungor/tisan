# Zero-Shot Zaman Serisi Anomali Modeli

Herhangi bir sektörden gelen sayısal zaman serisi verisinde, **hiç eğitim, etiket veya ayar gerektirmeden** anomali tespit eden açık kaynak bir model.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("kullanici/anomali-small", trust_remote_code=True)
y = model.predict(matris)      # ilk sütun timestamp, geri kalanlar değerler → (T,) 0/1 vektörü
sonuc = model.detect(matris)   # isteğe bağlı ayrıntı: skorlar, hangi sütun, olaylar
```

---

## 1. Amaç ve vizyon

**Hedef:** Hugging Face üzerinden dünya çapında kullanılan, "zaman serisi anomali tespiti denince akla gelen" model olmak. Tahmin (forecasting) alanında Chronos neyse, anomali tespitinde o olmak.

**Neden mümkün?**

- Bireysel olarak iz bırakan projeler (Moondream, MoritzLaurer'ın zero-shot sınıflandırıcıları, llama.cpp, QLoRA) genel amaçlı dev modeller değil, **tek bir işte en iyi olan** çalışmalardı.
- Tek bir H100 ile 10–200M parametrelik zaman serisi modelleri eğitilebilir. Chronos-Bolt modelleri 9M ile 205M parametre arasında.
- Elimizde 50 farklı sektörden veri erişimi var. Bireysel geliştiricilerin çoğunda olmayan asıl avantaj bu.

**Neden boşluk var?**

- Chronos, TimesFM ve Moirai **tahmin** modelleri. Tahmin hatasını anomali skoru olarak kullanmak yetersiz kalıyor: model kalıcı anomalileri de iyi tahmin ettiği için onları normal sanıyor (EUROCAST 2026 çalışması).
- Akademik anomali modelleri var (TimeRCD, DADA, TSPulse, MOMENT), ama hiçbiri Chronos kadar yaygın değil. Kurulumları zor, çıktıları yorumlaması güç.
- **Boşluk fikirde değil, kullanımda ve kolaylıkta.**

---

## 2. Temel tasarım kararları

| Karar | Gerekçe |
|---|---|
| Sadece zaman serisi, tablo verisi yok | Odak. Tablo verisi farklı mimari ve veri gerektiren ayrı bir problem |
| İlk sütun her zaman timestamp | Belirsizliği ortadan kaldırır; sıralama, boşluk tespiti ve tarihli çıktı sağlar |
| Takvim özellikleri yok, sadece Δt | Model döngüleri değerlerden öğrenir; alan ezberi zero-shot'ı bozar |
| Sütunlar her zaman 100'e tamamlanır | Sabit girdi boyutu; eksikler **sıfır + maske**, rastgele özellik değil |
| Sütun sırası eğitimde karıştırılır | Model pozisyon ezberlemesin, her sütunu davranışına göre değerlendirsin |
| Sütun adı istenmez | Model sütunun ne olduğuna değil, nasıl davrandığına bakar |
| Her pencere robust normalize edilir (medyan/MAD) | Ölçek bağımsızlığı: 20 °C ile 3000 TL aynı modelde çalışır |
| Tahmin değil, doğrudan anomali tespiti | Etiketli sentetik anomalilerle denetimli ön eğitim |
| Hücre bazında eğitim, satır bazında çıktı | Model hangi sütun olduğunu öğrenir; kullanıcı tek cevap alır |
| Kalibre skorlar | 0.9 gerçekten yaklaşık %90 demek olsun: temel farklılaşma noktası |
| pip paketi yok, HF remote code | Kullanıcı sadece modeli çağırıp matrisi verir |

---

## 3. Girdi ve çıktı

### Girdi

NumPy matrisi, şekli `(T, 1 + k)`:

- **1. sütun:** zaman damgası (unix saniye veya milisaniye, ISO tarih)
- **Diğer sütunlar:** 1 ile 100 arası sayısal değişken. 100'den fazlaysa 100'lük gruplar hâlinde işlenir.
- **Satır sayısı:** 20'den 5 milyona kadar. 20'nin altında kullanıcı uyarılır veya istatistiksel yönteme geçilir.

Arka planda otomatik yapılanlar: zaman biçimi tanıma, sıralama, tekrarlı zaman damgalarını birleştirme, frekans çıkarımı, boşluk tespiti, NaN işleme.

### Çıktı

**Birincil çıktı `predict()`:** satır başına 0/1. Model alanı bilmez ve bilmesi gerekmez; finans, sensör, siber güvenlik ya da hiç görülmemiş bir veri aynı yoldan geçer.

`detect()` ayrıntı isteyenler için:

```python
sonuc.anomaly_rows    # [4, 120, 121, 122]         hızlı bakış
sonuc.row_scores      # (T,)                       satır başına kalibre skor
sonuc.cell_scores     # (T, k)                     hangi sütun?
sonuc.events          # gruplanmış olaylar
sonuc.pattern         # sütun başına örüntü özeti
sonuc.to_dataframe()  # orijinal veri + skor + etiket
sonuc.plot()          # anomalileri işaretli grafik
```

Olay örneği:

```python
{"start": "2026-09-01 00:10", "end": "2026-09-01 00:10",
 "type": "spike", "channels": [2], "confidence": 0.98,
 "value": 58.4, "expected": 22.0,
 "reason": "Günlük döngüye göre ~22 bekleniyordu; 58.4 tipik aralığın çok dışında."}
```

Örüntü özeti (klasik istatistikle, NumPy ile hesaplanır):

```python
{"frequency": "5min", "seasonality": ["daily"], "trend": "flat",
 "volatility": "low", "typical_range": [20.8, 23.1]}
```

### Anomali türleri (isteğe bağlı, varsayılan kapalı)

`spike` · `level_shift` · `flatline` · `drift` · `noise_burst` · `pattern_change` · `correlation_break` · `missing_data`

Tür yalnızca sentetik etiketlerden öğrenilebilir (gerçek verinin türü yok) ve eğitim kapasitesini böler; `use_types=False` ile kapalı. Çıktı türü `"anomaly"`.

### Sorumluluk dağılımı

| Parça | Kim yapar |
|---|---|
| Anomali skoru ve türü | Sinir ağı (asıl zor iş) |
| Örüntü özeti | Klasik istatistik (otokorelasyon, Fourier) |
| Olay gruplama, eşikleme | Son işleme katmanı |
| Açıklama metni | Şablon (dil modeli yok, uydurma riski yok) |

---

## 4. Model mimarisi

```
Girdi (T, 1+k)
  → zaman sütunu ayrılır → Δt özelliği
  → k sütun robust normalize edilir, 100'e tamamlanır (0 + maske)
  → Patch gömme (16 nokta = 1 token, ağırlıklar tüm sütunlarda ortak)
  → [Zaman dikkati → Sütun dikkati] × N katman
  → Çıkış başlığı: (T, 100) anomali olasılığı (+ isteğe bağlı tür)
  → Maskeli sütunlar atılır
```

- **Zaman dikkati:** Her sütun kendi geçmişine bakar: "Bu sütun için normal ne?"
- **Sütun dikkati:** Aynı andaki sütunlar birbirine bakar: "Diğerleriyle tutarlı mı?" Sütunlara pozisyon bilgisi eklenmez; boş sütunlar maskelenir.
- İki ekseni ayırmak hesaplamayı tek H100'e sığdırır: 2048 satır / 16 = 128 token × 100 sütun.
- **Hedef boyut:** 10–50M parametre, bf16. İlk denemeler birkaç milyon parametreyle yapılır.
- **Aşamalı yol:** Önce kanal bağımsız çalıştığı kanıtlanır, sonra sütunlar arası ilişkiler güçlendirilir.
- **Konum kodlama (v4):** RoPE, konum = Δt oranlarının kümülatif toplamı (medyan adım = 1). Düzensiz örneklemede konum gerçek zamanı izler; kısa pencerede konum anlamı değişmez. Ablation: öğrenilmiş mutlak konum.
- **Çok ölçekli girdi (v4):** her sütun için 4 kanal: ham değer, birinci fark, 8 ve 64 satırlık yerel seviyeden sapma. Uzun bağlamdaki kayma ince patch'e girer.
- **Maskeli yeniden inşa (v4):** patch'lerin %30'u mask token ile gizlenip normalize değerler yeniden inşa edilir; etiketsiz 140M satırdan "normal" etiket olmadan öğrenilir (MOMENT/PatchTST). Önce ön eğitim, sonra anomali eğitimi (yardımcı kayıp olarak sürer). Yeniden inşa hatası ileride ikinci anomali sinyali.

### Eksik sütunlar ve ek kanallar

- Sütun sayısı 100'den azsa boş yerler **sıfır + maske** ile doldurulur; sütun dikkati ve kayıp fonksiyonu bu sütunları hiç görmez. Türetilmiş özelliklerle (hareketli ortalama, fark, FFT) doldurulmaz: bunlar mevcut sütunların fonksiyonudur, sütun dikkatini sahte korelasyonla bozar ve sütun sayısına göre değişen tutarsız bir girdi yaratır.
- Modele daha zengin girdi vermek için doğru yer **sütun başına ek kanal**: her patch'in gömme girdisine ham 16 değerin yanında 16 birinci fark eklenir (`extra_channels=["diff"]`). Sütun sayısı değişmez, kullanıcı tarafında aynı hesap otomatik yapılır. Açık/kapalı ablation ile karar verilir.
- Ön işleme (`robust_normalize`, `delta_t_feature`, dolgu, ek kanallar) tek bir `onisleme.py` modülünde durur; üretici, `Dataset` ve `detect()` hepsi oradan çağırır. Eğitim ile inference'ın ayrışması yapısal olarak engellenir.

---

## 5. Eğitim verisi

`egitim_verisi_uretici.py`, tek dosyada 16 gerçek alanı ve bir soyut alanı kapsar:

finans · imalat · uzay (uydu telemetrisi) · hasta takibi · EKG · glikoz (CGM) · biyoreaktör · enerji · bilişim · su şebekesi · çevre/iklim · perakende · otomotiv · telekom · tarım · kimyasal proses · soyut seriler

### Her alanda

- **Gerçekçi fizik:** Alana özgü ölçüm aralıkları (EKG 4 ms, borsa 1 gün), birimler, sınırlar (SpO2 ≤ 100, CPU 0–100), sensör hassasiyeti.
- **Normal ama tuhaf görünen davranışlar (etiketlenmez):** vardiya duruşları, uydu tutulmaları, yemek sonrası glikoz, GC testere dişi, sulama sıçramaları, set noktası değişimleri, hafta sonu borsa boşlukları. Yanlış alarmları azaltmanın anahtarı bu.
- **Alana özgü arıza senaryoları:** rulman aşınması, flash crash, bellek sızıntısı, ektopik atım, su kaçağı, kontaminasyon, termal kaçak... Birçoğu birden fazla sütunu tutarlı biçimde etkiler.

### Oranlar

| Ayar | Değer |
|---|---|
| Temiz (anomalisiz) örnek | %25 |
| Alana özgü / genel anomali | %70 / %30 |
| Veri boşluğu olan anomalili örnek | %12 |
| Sona kadar süren anomali | ~%37 |
| Sütun sayısı | 1–100, log-uniform (az sütun daha sık) |
| Satır sayısı | 20–2048 (yarısı tam pencere) |

### Kurallar

- Anomali gücü, serinin **kendi oynaklığına** göre ayarlanır.
- `difficulty` parametresi ile curriculum: önce belirgin, sonra ince anomaliler.
- Kalıcı anomalilerden önce en az %30 normal bağlam bırakılır.
- Veri boşlukları satır silinerek üretilir; zaman akmaya devam eder.
- Model alan adını **hiçbir zaman** görmez.

### Gerçek veri stratejisi

**Gerçek arka plan + sentetik anomali.** 50 sektörün gerçek verisi "normal"i öğretir, içine yerleştirilen sentetik anomaliler sınırsız ve bedava etiket sağlar.

- Tüm veri uzun formata çevrilir (`series_id, sector, timestamp, value`) ve Parquet'te saklanır.
- Gerçek verideki gizli anomaliler, robust istatistikle önceden ayıklanır.
- Sektörler dengeli örneklenir (büyük sektörler baskın çıkmasın).
- Veri lisansları yeniden kullanıma ve yayınlamaya uygun olmalı; özellikle finans verisinde dikkat.
- Eklemek için: `gen_real_template` doldurulur ve `DOMAINS` sözlüğüne eklenir.

### Veri havuzu (`data/`)

Gerçek veri `gercek_veri_hazirla.py` ile ortak formata çevrilir ve `data/havuz/` altında tutulur (ayrıntı: `data/README.md`). Veri dosyaları depoya girmez; sadece dönüştürme kodu paylaşılır.

| Kaynak | Seri | Etiket | Rol |
|---|---|---|---|
| SKAB, NAB, SMAP/MSL, SMD | ~300 | var | **Sadece değerlendirme.** TSB-AD ile çakışıyor; eğitime girerse benchmark sonuçları geçersiz olur |
| Pump sensor, CNC mill, SKAB teaser | 20 | kısmi | Eğitim arka planı (imalat) |
| LOTSA (105 alt küme, her birinden örnek) | ~10.000 | yok | Eğitim arka planı: ulaşım, enerji, iklim, bilişim, perakende, finans, sağlık |
| Kendi 50 sektör | – | yok | Asıl fark yaratacak kaynak, henüz eklenmedi |

**Kabul ölçütü:** Bir veri seti havuza girmeden önce rastgelelik testinden geçer: saat/gün dağılımı, saatlik hacimde otokorelasyon, olay aralıklarının dağılımı. Tamamen rastgele üretilmiş veride öğrenilecek "normal" yoktur (reddedilenler: Kaggle fraud işlemleri — düz dağılım, sıfır otokorelasyon, Poisson hacim; CIC-DDoS — 92 dakikalık akış tablosu; GlucoBench — nabız ve cilt sıcaklığı beyaz gürültü).

**Etiket güveni (eğitim):** `-1` etiketsiz hücreler "doğrulanmış normal" sayılmaz; düşük ağırlıklı (0.05) arka plan olarak girer.
C-MAPSS / IMS / FEMTO gibi zaman sınırından türetilen "bozulmaya yaklaşma" etiketleri zayıf etiket (0.2 ağırlık). Satır düzeyi etiketler
hücrelere yayılmaz; pozitif satırlarda kayıp satır skoru (sütunlar üzerinde max) üzerinden hesaplanır. Bkz. `training_labels.py`.

**Havuzdan eğitim örneği:** Pencere kesme havuza yazılmaz, eğitim sırasında rastgele yapılır: uzunluk 20–2048, sütun alt kümesi 1–100 (log-uniform), rastgele çözünürlük düşürme, anomalinin pencere içindeki konumu. Doğrulama seti sabit seed ile bir kez kesilir ve dosyaya yazılır.

---

## 6. Eğitim

- **Kayıp:** İkili çapraz entropi + focal loss (anomaliler nadir). Tür için çapraz entropi, sadece anomali noktalarında. Maskeli sütunlar kayba dahil edilmez.
- **Curriculum:** `difficulty` 0'dan 1'e.
- **Kalibrasyon:** Eğitim sonrası ayrı bir doğrulama setinde temperature scaling. Kısa pencereler de dahil edilir ki az bağlamda güven otomatik düşsün.
- **Araçlar:** PyTorch, NumPy, Polars, PyArrow, Accelerate, Weights & Biases.

---

## 7. Inference: farklı veri boyutları

| Durum | Yöntem |
|---|---|
| < 20 satır | Uyarı veya istatistiksel yönteme (MAD) geçiş |
| 20–2048 satır | Dolgu + maske; güven skoru düşer |
| 2048 satır | Tek pencere |
| Uzun veri | Kayan pencere (2048 / 1024 adım) + skorların birleştirilmesi |
| Çok uzun veri | Çok ölçekli işleme: orijinal, saatlik, günlük çözünürlük |

### Kalıcı anomaliler

Arıza haftalarca sürerse, sonraki pencerelerin tamamı anomali olur ve model onu "yeni normal" sanabilir. Önlemler:

- **Referans bağlam:** Her pencereye önceki normal bir bölüm eşlik eder.
- **Olay sürekliliği:** Anomaliyle biten pencereden sonra olay, veri normale dönene kadar açık kalır.
- **Kullanıcı kontrolü:** "Şu tarihten itibaren yeni normali kabul et" seçeneği.

---

## 8. Dağıtım (Hugging Face)

pip paketi yok. Kod ağırlıklarla birlikte HF deposunda durur.

```
kullanici/anomali-small/
├── config.json
├── model.safetensors
├── configuration_anomali.py   # PretrainedConfig
├── modeling_anomali.py        # PreTrainedModel + detect()
└── README.md                  # model kartı
```

```python
AnomaliConfig.register_for_auto_class()
AnomaliModel.register_for_auto_class("AutoModel")
model.push_to_hub("kullanici/anomali-small")
```

**Kurallar:**

- `detect()` tüm ön ve son işlemeyi içerir.
- Uzak kodda sadece `torch`, `numpy` ve `transformers` kullanılır. SciPy veya pandas yok.
- Sürümler etiketlenir, model kartında `revision="v1.0"` önerilir.
- Kod okunaklı tutulur; `trust_remote_code` güveni hak etmeli.

**Yayılma için:** Gradio ile HF Space demosu (CSV yükle, grafikte gör) ve iyi bir model kartı (3 satırlık kullanım, benchmark tablosu, desteklenen türler, sınırlar).

---

## 9. Değerlendirme

- **Zero-shot kanıtı:** Bazı sektörler eğitimden tamamen çıkarılır, sadece onlarda test edilir (leave-one-domain-out).
- **Benchmark'lar:** TSB-AD, UCR Anomaly Archive. Bunlar asla eğitimde kullanılmaz.
- **Metrikler:** VUS-PR gibi güvenilir metrikler. Sonuçları şişiren point-adjust kullanılmaz.
- **Rakipler:** TimeRCD, DADA, TSPulse, MOMENT.
- **Ablation:** Takvim özellikleri, sütun dikkati, model boyutu gibi kararlar deneyle test edilir.

---

## 10. Kim kullanır?

Asıl değer: **binlerce sensör veya metrik var, etiketli anomali verisi yok, her biri için model kuracak ekip yok.**

| Sektör | Çözdüğü problem |
|---|---|
| İmalat | Kademeli arızaların erken belirtileri, yeni makinelerde ilk günden izleme |
| Enerji | İnvertör, türbin ve trafo arızaları, kaçak tespiti |
| Bilişim (AIOps) | Binlerce metrikte sabit eşiklerin yerine akıllı alarm |
| Finans | Sistem sağlığı, işlem hacmi anomalileri, hatalı fiyat verisi |
| Telekom | Hücre kesintileri, uyuyan hücreler |
| Lojistik | Soğuk zincir sıcaklık takibi |
| Akıllı binalar | Bağlamsal tüketim anomalileri |

**Dolaylı ama çok değerli kitle:** Modeli kendi ürününe gömen izleme, IoT ve SCADA yazılım şirketleri.

**İlk hedef sektörler:** Sunucu/uygulama izleme (en hızlı benimseyen kitle) ve imalat sensörleri (en büyük ekonomik değer). Tanıtımda somut bir başarı hikâyesi, benchmark tablosundan daha ikna edicidir.

---

## 11. Sınırlar ve dürüst vaatler

- Model **geleceği tahmin etmez**; sapmayı veride göründüğü anda yakalar. Kademeli arızalarda bu çoğu zaman arızadan önce olur, ani arızalarda olmaz.
- Kalan ömür tahmini (RUL) kapsam dışı: etiketli arıza verisi gerektirir ve zero-shot'a aykırıdır.
- "Anomali" alana göre değişir: hissede %10 sıçrama normal olabilir, sensörde neredeyse kesin arızadır. Bunun için duyarlılık ayarı (`low / medium / high`) ve ileride isteğe bağlı geri bildirim.
- Az veriden (< 20 satır) güvenilir sonuç çıkmaz.
- Tanıtım cümlesi: *"Normalden sapmayı ilk göründüğü anda yakalar."* "Arızayı tahmin eder" değil.

---

## 12. Yol haritası

- [x] Problem tanımı, girdi/çıktı ve mimari tasarımı
- [x] Çok alanlı sentetik veri üreticisi (16 alan, alana özgü senaryolar)
- [x] Veri havuzu formatı, açık veri setleri (SKAB, NAB, SMAP/MSL, SMD, Pump, CNC, LOTSA)
- [ ] 50 sektör verisinin havuza eklenmesi
- [x] Ortak ön işleme (`hf_model/modeling_anomali.py`: eğitim ve `detect()` aynı kodu kullanır)
- [x] Eğitim akışı (`egitim.ipynb`): havuz + sentetik, rastgele uzunluk/sütun/çözünürlük, curriculum, focal loss
- [x] Model iskeleti (iki eksenli dikkat, `extra_channels` ayarı, 6.4M parametre)
- [x] GPU'da ilk eğitimler (v1 6.4M, v2 6.4M + yeni veri, v3 34M): sentetikte iyi, gerçekte AUC-ROC ~0.65 tavanı
- [x] Eğitim akışı denetimi (7 madde): etiket güveni (etiketsiz ≠ normal, zayıf etiket düşük ağırlık), satır/hücre kaybı ayrımı,
      geçerli enjeksiyon (uygunluk + etki + çakışma kontrolü), worker'lara ulaşan curriculum, referanslı normalizasyon
      (eğitimde %25, inference'ta dondurulabilir otomatik referans), model seçimi / kalibrasyon / test için ayrı kaynaklar,
      satır düzeyinde kalibrasyon
- [x] Mimari v4: RoPE + gerçek zaman konumu, çok ölçekli girdi, maskeli yeniden inşa ön eğitimi
- [ ] v4 eğitimi (34M); ablation: ön eğitim açık/kapalı, çok ölçekli açık/kapalı, RoPE/learned, sentetik payı, leave-one-domain-out
- [ ] Benchmark'larda rakiplerle karşılaştırma
- [x] Kalibrasyon (temperature scaling, defterde)
- [x] `detect()`: ön/son işleme, kayan pencere, sütun gruplama, olaylar (çok ölçek ve örüntü özeti v2)
- [x] HF'ye yükleme (remote code) ve model kartı (defterde)
- [ ] Gradio Space demosu
- [ ] v0.1 yayını, blog yazısı / arXiv makalesi, geri bildirim toplama
- [ ] v2: güçlü çok değişkenli ilişkiler, akış (streaming) modu, sağlık skoru

**İlke:** Mükemmeli bekleme, erken yayınla. İlk 10 gerçek kullanıcının geri bildirimi, modelin kendisinden daha çok şey öğretir.

---

## 13. Dosyalar

| Dosya | İçerik |
|---|---|
| `README.md` | Bu belge: hedefler, kararlar, tasarım, yol haritası |
| `egitim_verisi_uretici.py` | Çok alanlı eğitim verisi üreticisi. Çalıştırınca istatistik, örnek CSV ve alan panosu üretir |
| `ornek_egitim_verisi.csv` | Kullanıcı biçiminde örnek: timestamp, değerler, satır bazında `anomali`, `tur`, `sutunlar` |
| `alan_ornekleri.png` | Her alandan bir eğitim örneği ve enjekte edilen anomaliler |
| `gercek_veri_hazirla.py` | Gerçek veri setlerini (`data/raw`) ortak havuz formatına çevirir; `seri_yukle()` ile okunur |
| `data/` | Veri havuzu: gerçek ve sentetik veri, katalog (bkz. `data/README.md`); HF kopyası `cagrigungor/tisan-havuz` |
| `hf_model/` | HF uzak kod: `configuration_anomali.py`, `modeling_anomali.py` (ortak ön işleme + iki eksenli dikkat + `detect()`) |
| `egitim.ipynb` | Eğitim defteri: HF havuzu + sentetik → eğitim, kalibrasyon, benchmark, HF'ye yükleme. Colab'da GPU ile çalışır |

```bash
python egitim_verisi_uretici.py
```

```python
import numpy as np
from egitim_verisi_uretici import make_sample

ornek = make_sample(np.random.default_rng(0), difficulty=0.5)
ornek["raw"]       # (T, 1+k)   kullanıcının vereceği matris
ornek["values"]    # (2048, 100) model girdisi
ornek["labels"]    # (2048, 100) hücre etiketleri
ornek["types"]     # (2048, 100) anomali türleri
ornek["meta"]      # alan, aralık, sütun rolleri
```
