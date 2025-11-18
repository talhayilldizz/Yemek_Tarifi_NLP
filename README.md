#  Akıllı Yemek Tarif Asistanı

**Türk yemek veri seti + Kendi geliştirdiğim Linear SVM Modeli + GPT-4o-mini Entegrasyonu**

Bu proje, kullanıcıdan alınan malzemeleri analiz ederek **hangi Türk yemeğinin yapılabileceğini tahmin eden** ve bu tahmin üzerinden **akıllı bir tarif oluşturan** uçtan uca bir yapay zeka uygulamasıdır.

Sistem hibrit bir yapıda çalışır:
1.  **Sınıflandırma (Classification):** Kendi eğittiğim **Linear SVM** modeli, girilen malzemelere göre en olası Türk yemeğini tahmin eder.
2.  **Üretim:** Tahmin edilen yemek ismi OpenAI **GPT-4o-mini** modeline gönderilir ve kullanıcı için kısa, uygulanabilir bir tarif oluşturulur.

---

## 📂 Veri Seti 

Model eğitimi için kullanılan veri seti Kaggle üzerinden alınmış, proje hedeflerine göre filtrelenmiş ve işlenmiştir.

* **Kaynak:** [Kaggle – Recipes of Countries](https://www.kaggle.com/datasets/kadirkdr/recipes-of-countries)
---

## 🧠 Model Mimarisi: Linear SVM

Bu projenin sınıflandırma katmanı, hazır bir API değil, **tamamen tarafımdan geliştirilen** bir makine öğrenmesi modelidir.

### Geliştirme Adımları:
-   **Veri Temizliği:** Malzeme listesindeki gürültülü verilerin ayıklanması.
-   **Vektörleştirme:** Metin verilerinin **TF-IDF** yöntemi ile sayısal vektörlere dönüştürülmesi.
-   **Model Eğitimi:** `LinearSVC` algoritması kullanılarak modelin eğitilmesi.
-   **Optimizasyon:** Hiperparametre ayarlamaları ve `foodname_map.json` ile yemek isimlerinin normalizasyonu.

📁 **Eğitilen Model:** `models/model3_linear_svc.pkl`

---

##  Çalışma Mantığı (Pipeline)

1.   **Girdi:** Kullanıcı elindeki malzemeleri web arayüzüne yazar.
2.   **İşleme:** Metin temizlenir ve TF-IDF vektörleştirici ile sayısallaştırılır.
3.   **Tahmin (SVM):** Eğittiğim Linear SVM modeli, bu malzemelerle yapılabilecek en uygun yemeği tahmin eder.
4.   **Üretim (GPT-4o-mini):** Tahmin edilen yemek ismi GPT modeline prompt olarak gönderilir.
5.   **Sonuç:** GPT, yemek için kısa ve anlaşılır bir tarif oluşturur.
6.   **Arayüz:** Sonuçlar HTML arayüzünde kullanıcıya sunulur.

---

##  Kullanılan Teknolojiler

* **Backend:** FastAPI / Flask
* **Makine Öğrenmesi:** scikit-learn (Linear SVM), pandas, numpy
* **NLP:** TF-IDF Vectorizer
* **LLM:** OpenAI GPT-4o-mini
* **Frontend:** HTML, CSS, JavaScript

---

##  Proje Yapısı

```text
Yemek_Tarifi_NLP/
│
├── app.py                   # Ana Uygulama (API: SVM Tahmini + GPT Entegrasyonu)
├── static/
│   └── index.html           # Kullanıcı Web Arayüzü
│
├── models/
│   ├── model3_linear_svc.pkl # Eğitilmiş Linear SVM Modeli
│   └── vectorizer_3.pkl      # TF-IDF Vektörleştirici
│
├── datasets/
│   ├── tr_yemekler_temiz.csv       # Temizlenmiş Veri
│   ├── tr_yemekler_arttirilmis.csv # Artırılmış Veri
│   └── foodname_map.json           # Yemek Adı Eşleştirme Haritası
│
├── .env                     # API Anahtarları (Gizli Dosya)
└── README.md                # Proje Dokümantasyonu
```

##  Kurulum
**1. Sanal Ortam Oluşturma:**
  python -m venv venv
  venv\Scripts\activate


**2. Gereksinimleri Yükleme:**
  pip install pandas fastapi uvicorn pydantic joblib openai python-dotenv scikit-learn

**3. .env Dosyasını Ayarlama:**
  Proje ana dizininde .env dosyası oluşturun ve API anahtarınızı ekleyin

**4. Uygulamayı Başlatma:**
  python app.py
