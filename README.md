# MTLE Kuantum Projesi 
 
Bu repo, Mezyal Temporal Lob Epilepsisi (MTLE) tanısı için geliştirilmiş hibrit kuantum-klasik snflandrma modelini i‡erir. 
 
## ?? Nihai Sonuçlar 
| Sınıflandırıcı | Doğruluk (Ort ñ SS) | 
|----------------|---------------------| 
| SVM (RBF)       | 0.97 ñ 0.03         | 
| Random Forest   | 0.96 ñ 0.02         | 
| VQC (6-qubit)   | 0.81 ñ 0.00         | 
 
## ?? €alŸtrma 
```bash 
pip install -r requirements.txt 
python mtle_hizli_son.py 
``` 
 
## ?? Dosyalar 
- `mtle_hizli_son.py`: Son ‡alŸan hzlandrlmŸ versiyon 
- `qbfinal.py`: €oklu hasta destekli ana versiyon 
- `results.txt`: Sonu‡lar 
- `requirements.txt`: Gerekli ktphaneler 
