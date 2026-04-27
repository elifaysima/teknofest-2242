# MTLE Kuantum Projesi 🧠⚛️

Bu repo, Mezyal Temporal Lob Epilepsisi (MTLE) tanısı için geliştirilmiş hibrit kuantum-klasik sınıflandırma modelini içerir.

## 📊 Nihai Sonuçlar
| Sınıflandırıcı | Doğruluk (Ort ± SS) |
|----------------|---------------------|
| SVM (RBF)      | 0.97 ± 0.03         |
| Random Forest  | 0.96 ± 0.02         |
| VQC (6‑qubit)  | 0.81 ± 0.00         |

## 📁 Dosyalar
- `mtle_hizli_son.py`: Son çalışan hızlandırılmış versiyon
- `results.txt`: Sonuçlar
- `requirements.txt`: Gerekli kütüphaneler

## ⚙️ Detaylı Çalıştırma Adımları

1. **Python Ortamını Hazırlayın (Anaconda önerilir)**
   ```bash
   conda create -n mtle_env python=3.10 -y
   conda activate mtle_env
   ```

2. **Gerekli Kütüphaneleri Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **Veri Setlerini İndirin**
   Model, aşağıdaki açık erişimli iEEG veri setlerini kullanır. Hepsini masaüstünüze indirin:
   - ds003029 (Epilepsy Multicenter): https://openneuro.org/datasets/ds003029
   - ds004993 (B(RAIN)2): https://openneuro.org/datasets/ds004993
   - ds003374 (Amygdala iEEG): https://openneuro.org/datasets/ds003374

   İndirme komutu (her biri için):
   ```bash
   openneuro-py download --dataset=ds003029
   openneuro-py download --dataset=ds004993 --include=sub-W1
   openneuro-py download --dataset=ds003374 --include=sub-01
   ```

4. **Klasör Yollarını Ayarlayın**
   `mtle_hizli_son.py` dosyasının başındaki `BASE_DIRS` sözlüğünü, veri setlerini indirdiğiniz klasörlere göre düzenleyin:
   ```python
   BASE_DIRS = {
       "epilepsy":  "C:/Users/KULLANICI_ADI/Desktop/ds003029",
       "control_BRAIN2": "C:/Users/KULLANICI_ADI/Desktop/ds004993",
       "control_amygdala": "C:/Users/KULLANICI_ADI/Desktop/ds003374"
   }
   ```

5. **Veri Setlerinin Tam Olduğunu Kontrol Edin**
   En az bir hasta ve bir kontrol bireyi olduğundan emin olun.

6. **Modeli Çalıştırın**
   ```bash
   python mtle_hizli_son.py
   ```
   Sonuçlar 5‑katlı çapraz doğrulama ile hesaplanacak ve ekrana yazdırılacaktır.

## 🧪 Tekrar Edilebilirlik
Tüm deneyler aynı rastgele tohum (`random_state=42`) ile gerçekleştirilmiştir. Aynı sonuçları almak için veri setlerini belirtilen sürümlerle kullanın.

## 🖋️ Makale
Bu çalışmanın 15 sayfalık tam metni için `docs/` klasörüne bakın.
