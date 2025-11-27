# ⚽ HalısahaAI - Profesyonel Maç Analiz Sistemi

HalısahaAI, amatör futbol maçlarını yapay zeka ile analiz ederek profesyonel bir deneyime dönüştüren bir web uygulamasıdır. YOLOv8 görüntü işleme teknolojisi kullanarak oyuncuları takip eder, istatistikler çıkarır ve maçın hikayesini yazar.

![HalısahaAI Arayüzü](https://images.unsplash.com/photo-1579952363873-27f3bde9be2e?w=800&q=80)

## 🌐 Canlı Demo
[https://halisahaai.netlify.app/](https://halisahaai.netlify.app/)

## 🚀 Özellikler

*   **Yapay Zeka Destekli Analiz:** YOLOv8 ve OpenCV kullanarak oyuncu takibi, koşu mesafesi ve hız analizi.
*   **Detaylı İstatistikler:** Her oyuncu için Hız, Şut, Pas, Dribling, Defans ve Fizik (OVR) puanları.
*   **Maç Özeti & Hikaye:** Otomatik maç skoru, "Maçın Adamı" (MVP) seçimi ve dinamik canlı anlatım.
*   **Görsel Şölen:**
    *   **Radar Scan:** Analiz sırasında oyuncu tarama animasyonu.
    *   **Kadro Görünümü:** Saha dizilişi ve oyuncu kartları.
    *   **Galatasaray Modu:** Oyuncu yüzleri algılanamazsa Galatasaray yıldızlarının (Icardi, Muslera vb.) görselleri kullanılır.
*   **Video İşleme:**
    *   Kendi maç videonuzu yükleyin.
    *   YouTube linki yapıştırın (Otomatik indirme ve analiz).

## 🛠️ Teknolojiler

*   **Backend:** Python, FastAPI, Uvicorn, YOLOv8 (Ultralytics), OpenCV, NumPy, Scikit-learn.
*   **Frontend:** HTML5, React (CDN), TailwindCSS (CDN).
*   **Veri İşleme:** K-Means Clustering (Takım ayrıştırma), yt-dlp (YouTube indirme).

## 📦 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler
*   Python 3.9 veya üzeri
*   Node.js (Opsiyonel, sadece frontend sunucusu için)

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/kullaniciadi/halisaha-ai.git
cd halisaha-ai
```

### 2. Backend Kurulumu (Python)
Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
*Eğer `requirements.txt` yoksa:* `pip install fastapi uvicorn opencv-python numpy ultralytics scikit-learn python-multipart yt-dlp`

Backend sunucusunu başlatın:
```bash
python main.py
```
Sunucu `http://localhost:8000` adresinde çalışacaktır.

### 3. Frontend Kurulumu
Frontend tek bir `index.html` dosyasından oluşur. Doğrudan tarayıcıda açabilir veya bir yerel sunucu kullanabilirsiniz:

```bash
# Node.js ile (Tavsiye edilen)
npx http-server .
```
Tarayıcınızda `http://localhost:8080` adresine gidin.

## 🌍 Deployment (Yayına Alma)

Bu proje **Frontend** ve **Backend** olmak üzere iki parçadan oluşur.

1.  **Frontend (Netlify/Vercel):** `index.html` dosyası statik olarak Netlify veya Vercel üzerinde barındırılabilir.
2.  **Backend (Render/Railway):** Python API sunucusu (main.py) GPU destekli veya yüksek işlem gücüne sahip bir sunucuda çalışmalıdır (Örn: Render, Railway, AWS).

*Not: Sadece Frontend'i Netlify'a yüklerseniz, Backend yerel bilgisayarınızda çalışıyorsa (localhost), uygulama sadece sizin bilgisayarınızda çalışır.*

## 🤝 Katkıda Bulunma
Pull request'ler kabul edilir. Büyük değişiklikler için lütfen önce tartışma başlatın.

## 📄 Lisans
MIT License
