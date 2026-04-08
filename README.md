🌙 Yaşam Tarzı Verileri Kullanılarak Makine Öğrenmesi Modelleri İle Uyku Bozukluklarının Sınıflandırması
Bu proje, bireylerin günlük yaşam alışkanlıklarını ve fizyolojik verilerini analiz ederek Uyku Apnesi ve İnsomnia (Uykusuzluk) gibi bozuklukları makine öğrenmesi yöntemleriyle tahmin etmeyi amaçlar.

🚀 Proje Hakkında
Geleneksel teşhis yöntemleri (Polisomnografi gibi) maliyetli ve zaman alıcı olduğu için, bu projede yapay zeka desteğiyle ön teşhis koyabilecek bir model geliştirilmiştir. Proje kapsamında veri ön işleme, özellik mühendisliği (Feature Engineering) ve veri dengeleme teknikleri uygulanmıştır.

📊 Model Başarısı
En iyi performansı gösteren XGBoost modeli ile elde edilen sonuçlar:

Doğruluk (Accuracy): %91.95

ROC-AUC: 0.9189

🛠️ Kullanılan Teknolojiler
Programlama: Python

Veri Analizi: Pandas, Numpy

Makine Öğrenmesi: Scikit-learn, XGBoost, LightGBM

Açıklanabilir YZ (XAI): SHAP

Arayüz: Gradio

📁 Dosya Yapısı
app.py: Gradio tabanlı kullanıcı arayüzü.

bitirme_projesi.ipynb: Veri analizi, görselleştirme ve model eğitim süreci.

sleep_model.joblib: Eğitilmiş XGBoost modeli.

requirements.txt: Gerekli kütüphaneler listesi.

veri_seti.csv: Eğitimde kullanılan veri seti.

📈 Analiz ve Görselleştirme
Modelin kararlarını ve performansını anlamak için kullanılan temel grafikler:

1. Karışıklık Matrisi (Confusion Matrix)
Modelin hangi sınıfları ne kadar doğru tahmin ettiğini gösterir.

2. SHAP Analizi (Model Açıklanabilirliği)
Modelin tahmin yaparken hangi özelliklere (Yaş, Kan Basıncı, Stres Seviyesi vb.) ne kadar önem verdiğini gösterir.

💻 Kurulum ve Çalıştırma
Projeyi klonlayın:

Bash
git clone https://github.com/kullaniciadi/proje-adi.git
Gerekli kütüphaneleri yükleyin:

Bash
pip install -r requirements.txt
Uygulamayı başlatın:

Bash
python app.py
