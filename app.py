import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io
import base64
import os

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Diyabetik Retinopati Teşhis Sistemi",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
        font-weight: 500;
    }
    .info-box {
        background: #e8f4fd;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #004085;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    """Model yükleme - cache ile optimize edilmiş"""
    try:
        # Model dosyası yolu - kendi yolunuzu buraya yazın
        model_path = "model/dr_model.h5"  # Model dosyanızın yolunu buraya yazın
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model dosyası bulunamadı: {model_path}")
            st.info("💡 Model dosyanızı 'model' klasörüne koyun veya yolu güncelleyin")
            return None
            
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        return None

# Görüntü ön işleme
def preprocess_image(image):
    """Yüklenen görüntüyü model için hazırlar ve ön işleme adımlarını gösterir"""
    try:
        # PIL Image'ı numpy array'e çevir
        img_array = np.array(image)
        
        # RGB formatında olduğundan emin ol
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Orijinal boyutları al
        height, width = img_rgb.shape[:2]
        
        # 1. Dairesel maske oluştur (Circular Masking)
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 2 - 10
        cv2.circle(mask, center, radius, 255, -1)
        
        # Maskeyi uygula
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # 2. CLAHE ile kontrast geliştirme
        lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 3. Bilateral filtreleme (Gürültü azaltma)
        denoised_img = cv2.bilateralFilter(enhanced_img, 9, 75, 75)
        
        # 4. 224x224'e yeniden boyutlandır
        img_resized = cv2.resize(denoised_img, (224, 224))
        
        # 5. VGG16 preprocessing
        img_preprocessed = preprocess_input(img_resized)
        
        # Batch dimension ekle
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Ön işleme adımlarını return et
        steps = {
            'original': cv2.resize(img_rgb, (224, 224)),
            'masked': cv2.resize(masked_img, (224, 224)),
            'enhanced': cv2.resize(enhanced_img, (224, 224)),
            'final': img_resized
        }
        
        return img_batch, steps
    except Exception as e:
        st.error(f"Görüntü işlenirken hata: {str(e)}")
        return None, None

# Tahmin fonksiyonu
def predict_image(model, processed_image):
    """Model ile tahmin yapar"""
    if model is None:
        return None
    
    try:
        prediction = model.predict(processed_image, verbose=0)
        return prediction
    except Exception as e:
        st.error(f"Tahmin yapılırken hata: {str(e)}")
        return None

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">
        🔬 Diyabetik Retinopati Teşhis Sistemi
    </h1>
    <p style="color: white; text-align: center; margin: 0; font-size: 18px;">
        VGG16 Transfer Learning ile Geliştirilen AI Tabanlı Tanı Sistemi
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Proje bilgileri
with st.sidebar:
    st.markdown("### 📋 Proje Detayları")
    
    # Proje istatistikleri
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Doğruluk", "77%")
    with col2:
        st.metric("F1 Score", "77%")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Sınıf", "5")
    with col4:
        st.metric("Test", "131")
    
    st.markdown("---")
    st.markdown("**Model:** VGG16 Transfer Learning")
    st.markdown("**Dataset:** IDRID")
    st.markdown("**Framework:** TensorFlow/Keras")
    
    st.markdown("---")
    st.markdown("###  Sınıflar")
    st.markdown("• **Grade 0:** Normal")
    st.markdown("• **Grade 1:** Hafif DR")
    st.markdown("• **Grade 2:** Orta DR") 
    st.markdown("• **Grade 3:** Şiddetli DR")
    st.markdown("• **Grade 4:** Proliferatif DR")
    
    st.markdown("---")
    st.info("⚠️ Bu sistem tanı amaçlı değil, yardımcı araç olarak geliştirilmiştir.")

# Model yükleme
model = load_model()

# Ana içerik alanı
if model is not None:
    st.success("✅ Model başarıyla yüklendi!")
else:
    st.error("❌ Model yüklenemedi. Lütfen model dosyası yolunu kontrol edin.")

# Ana uygulama
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Görüntü Yükleme")
    
    # Dosya yükleme
    uploaded_file = st.file_uploader(
        "Fundus görüntüsünü seçin",
        type=['png', 'jpg', 'jpeg'],
        help="Retina fundus görüntüsü yükleyin (PNG, JPG, JPEG formatları desteklenir)"
    )
    
    # Örnek görüntüler
    st.markdown("### 📸 Örnek Görüntüler")
    st.info("💡 Test için örnek fundus görüntülerini internetten indirip deneyebilirsiniz.")
    
    # Demo butonu
    if st.button("🎮 Demo Veri ile Test Et"):
        # Rastgele demo verisi simülasyonu
        demo_data = {
            'class': ['Normal', 'Hafif DR', 'Orta DR', 'Şiddetli DR', 'Proliferatif DR'],
            'probability': [85.2, 8.3, 3.1, 2.8, 0.6]
        }
        
        with col2:
            st.header("📊 Demo Sonucu")
            st.success("✅ **Normal (Grade 0)**")
            st.metric("Güven Oranı", "85.2%")
            
            # Demo grafik
            fig = px.bar(
                x=demo_data['class'],
                y=demo_data['probability'],
                title="Demo - Sınıf Olasılıkları (%)",
                color=demo_data['probability'],
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

# Görüntü analizi
if uploaded_file is not None:
    with col1:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
        
        # Görüntü bilgileri
        st.markdown("### 📏 Görüntü Bilgileri")
        st.write(f"**Boyut:** {image.size}")
        st.write(f"**Format:** {image.format}")
        st.write(f"**Mod:** {image.mode}")
    
    # Analiz butonu
    if st.button("🔍 Analiz Et", type="primary") and model is not None:
        with st.spinner("🔬 Görüntü analiz ediliyor..."):
            # Görüntüyü işle
            processed_img, preprocessing_steps = preprocess_image(image)
            
            if processed_img is not None:
                # Tahmin yap
                prediction = predict_image(model, processed_img)
                
                if prediction is not None:
                    with col2:
                        st.header("📊 Analiz Sonucu")
                        
                        # Sınıf isimleri
                        class_names = [
                            'Normal (Grade 0)',
                            'Hafif DR (Grade 1)', 
                            'Orta DR (Grade 2)',
                            'Şiddetli DR (Grade 3)',
                            'Proliferatif DR (Grade 4)'
                        ]
                        
                        # Olasılıkları hesapla
                        probabilities = prediction[0] * 100
                        max_idx = np.argmax(probabilities)
                        max_prob = probabilities[max_idx]
                        predicted_class = class_names[max_idx]
                        
                        # Sonucu göster - Daha görünür renklerle
                        if max_idx == 0:
                            st.markdown(f"""
                            <div style="background: #d4edda; border: 2px solid #c3e6cb; border-radius: 8px; padding: 15px; margin: 10px 0;">
                                <h3 style="color: #155724; margin: 0;">✅ {predicted_class}</h3>
                                <p style="color: #155724; margin: 5px 0;">Retinada diyabetik retinopati belirtisi görülmüyor.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 15px; margin: 10px 0;">
                                <h3 style="color: #721c24; margin: 0;">⚠️ {predicted_class}</h3>
                                <p style="color: #721c24; margin: 5px 0;">Diyabetik retinopati tespit edildi. Göz doktoruna başvurun.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Güven oranı
                        st.metric("🎯 Güven Oranı", f"{max_prob:.1f}%")
                        
                        # Olasılık grafiği
                        fig = px.bar(
                            x=class_names,
                            y=probabilities,
                            title="Sınıf Olasılıkları (%)",
                            color=probabilities,
                            color_continuous_scale="viridis",
                            labels={'x': 'Sınıflar', 'y': 'Olasılık (%)'}
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detaylı sonuçlar
                        st.markdown("### 📈 Detaylı Sonuçlar")
                        results_df = pd.DataFrame({
                            'Sınıf': class_names,
                            'Olasılık (%)': [f"{p:.2f}" for p in probabilities]
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                        # İşlenmiş görüntü adımları
                        with st.expander("🔬 Ön İşleme Adımları"):
                            st.markdown("**Görüntü işleme pipeline'ı:**")
                            
                            # 4 sütunlu düzen
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                st.image(preprocessing_steps['original'], 
                                        caption="1. Orijinal (224x224)", 
                                        use_container_width=True)
                            
                            with col_b:
                                st.image(preprocessing_steps['masked'], 
                                        caption="2. Dairesel Maskeleme", 
                                        use_container_width=True)
                            
                            with col_c:
                                st.image(preprocessing_steps['enhanced'], 
                                        caption="3. CLAHE Kontrast", 
                                        use_container_width=True)
                            
                            with col_d:
                                st.image(preprocessing_steps['final'], 
                                        caption="4. Bilateral Filtreleme", 
                                        use_container_width=True)
                            
                            st.info("🔍 **Açıklama:** Orijinal görüntü sırasıyla dairesel maskeleme, CLAHE kontrast geliştirme ve bilateral filtreleme adımlarından geçirilmiştir.")

# Model performansı bölümü
st.markdown("---")
st.header("📈 Model Performansı")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">77%</h3>
        <p style="margin: 0;">Genel Doğruluk</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">77%</h3>
        <p style="margin: 0;">Weighted F1</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">131</h3>
        <p style="margin: 0;">Test Örnekleri</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin: 0;">5</h3>
        <p style="margin: 0;">Sınıf Sayısı</p>
    </div>
    """, unsafe_allow_html=True)

# Sınıf bazında performans
st.markdown("### 🎯 Sınıf Bazında Performans")

performance_data = {
    'Sınıf': ['Normal', 'Hafif DR', 'Orta DR', 'Şiddetli DR', 'Proliferatif DR'],
    'Precision': [0.76, 0.86, 0.68, 0.68, 0.88],
    'Recall': [0.76, 0.89, 0.52, 0.85, 0.81],
    'F1-Score': [0.76, 0.87, 0.59, 0.75, 0.85],
    'Support': [25, 27, 25, 27, 27]
}

df = pd.DataFrame(performance_data)

# Performans tablosu
st.dataframe(df, use_container_width=True)

# Performans grafikleri
col1, col2 = st.columns(2)

with col1:
    # Precision grafik
    fig_precision = px.bar(
        df, x='Sınıf', y='Precision',
        title='Sınıf Bazında Precision',
        color='Precision',
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig_precision, use_container_width=True)

with col2:
    # F1-Score grafik
    fig_f1 = px.bar(
        df, x='Sınıf', y='F1-Score',
        title='Sınıf Bazında F1-Score',
        color='F1-Score',
        color_continuous_scale='greens'
    )
    st.plotly_chart(fig_f1, use_container_width=True)

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; padding: 25px; margin: 20px 0;">
    <h4 style="color: #343a40; margin-bottom: 15px;">ℹ️ Önemli Bilgiler</h4>
    <ul style="color: #495057; font-size: 14px; line-height: 1.6;">
        <li><strong>Model Tipi:</strong> VGG16 Transfer Learning</li>
        <li><strong>Eğitim Verisi:</strong> IDRID Dataset</li>
        <li><strong>Görüntü Boyutu:</strong> 224x224 piksel</li>
        <li><strong>Ön İşleme:</strong> CLAHE, bilateral filtering, circular masking</li>
        <li><strong>Regülarizasyon:</strong> Dropout (0.5) + L2 regularization</li>
    </ul>
    
    <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; margin-top: 15px;">
        <p style="color: #856404; margin: 0; font-weight: 600;">
            <strong>⚠️ Yasal Uyarı:</strong> Bu sistem yalnızca eğitim ve araştırma amaçlıdır. 
            Klinik tanı için kullanılmamalıdır. Kesin tanı ve tedavi için göz doktoruna başvurun.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎓 Biyomedikal Mühendisliği Bitirme Projesi | 
    🔬 Derin Öğrenme ile Diyabetik Retinopati Teşhisi</p>
</div>
""", unsafe_allow_html=True)
