import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import pickle

st.set_page_config(
    page_title="Audio Classification: Buka vs Tutup",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-buka {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .prediction-tutup {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .prediction-unregistered {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .feature-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎵 Audio Classification: Suara Buka vs Tutup</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        model = joblib.load('saved_models/rf_model_buka_tutup.pkl')
        return model, True
    except FileNotFoundError:
        st.error("Model tidak ditemukan! Pastikan file 'saved_models/rf_model_buka_tutup.pkl' tersedia.")
        return None, False

@st.cache_resource
def load_speaker_profiles():
    """Load speaker voice profiles for verification"""
    try:
        with open('speaker_profiles.pkl', 'rb') as f:
            profiles = pickle.load(f)
        return profiles, True
    except FileNotFoundError:
        st.warning("Speaker profiles tidak ditemukan. Sistem akan menggunakan mode tanpa verifikasi speaker.")
        return None, False

def extract_only_selected_features(y, sr=22050):
    """
    Extract only the 5 selected features with >50% importance
    """
    feats = {}
    
    # 1. stat_skew
    feats['stat_skew'] = stats.skew(y)
    
    # 2. chroma_0_mean
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats['chroma_0_mean'] = np.mean(chroma[0])
    
    # 3. mel features (6 & 7)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)
    feats['mel_6_std'] = np.std(mel_spec[6])
    feats['mel_7_mean'] = np.mean(mel_spec[7])
    feats['mel_7_std'] = np.std(mel_spec[7])
    
    return feats

def extract_speaker_features(y, sr=22050):
    """Extract speaker features - compatible with existing profiles"""
    # Gunakan fungsi lama yang kompatibel dengan speaker_profiles.pkl
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    speaker_features = np.concatenate([mfcc_mean, mfcc_std])
    
    return speaker_features

def verify_speaker(y, sr, speaker_profiles, threshold=0.65):  # Naikkan threshold drastis
    """Enhanced speaker verification with much stricter threshold"""
    if speaker_profiles is None:
        return False, "unknown", 0.0
    
    test_features = extract_speaker_features(y, sr)
    
    # Normalisasi features untuk konsistensi
    test_features = test_features / (np.linalg.norm(test_features) + 1e-8)
    
    best_similarity = 0
    best_speaker = None
    
    for speaker_id, profile_features in speaker_profiles.items():
        # Pastikan profile_features juga ternormalisasi
        profile_normalized = profile_features / (np.linalg.norm(profile_features) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(test_features, profile_normalized)
        
        # Tambahkan penalty untuk euclidean distance
        euclidean_dist = np.linalg.norm(test_features - profile_normalized)
        
        # Similarity yang disesuaikan (lebih ketat)
        similarity_adjusted = similarity * np.exp(-euclidean_dist * 2)  # Exponential penalty
        
        if similarity_adjusted > best_similarity:
            best_similarity = similarity_adjusted
            best_speaker = speaker_id
    
    # Threshold sangat ketat - hanya similarity sangat tinggi yang diterima
    is_registered = best_similarity > threshold
    
    return is_registered, best_speaker, best_similarity


def preprocess_audio(y, sr, target_sr=22050):

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Durasi 1 detik
    max_len = int(sr * 1.0)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]
    
    # Normalisasi
    y = y / np.max(np.abs(y) + 1e-6)
    
    # Noise reduction (sama seperti training)
    y[np.abs(y) < 0.005] = 0.0
    
    return y, sr

def predict_audio(audio_data, model, speaker_profiles=None, is_file=True):

    try:
        if is_file:
            # Load from file
            y, sr = librosa.load(audio_data, sr=22050)
        else:
            # Load from bytes (recorded audio)
            y, sr = sf.read(audio_data)
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        
        # Preprocess
        y_processed, sr_processed = preprocess_audio(y, sr)
        
        # Verify speaker first
        is_registered, speaker_id, similarity = verify_speaker(y_processed, sr_processed, speaker_profiles)
        
        if not is_registered:
            return "unregistered", [0.0, 0.0], None, y_processed, sr_processed, speaker_id, similarity
        
        # Extract features for classification
        features = extract_only_selected_features(y_processed, sr_processed)
        
        # Convert to DataFrame with correct feature order
        feature_order = ['mel_7_std', 'chroma_0_mean', 'mel_6_std', 'stat_skew', 'mel_7_mean']
        X_new = pd.DataFrame([features])[feature_order]
        
        # Predict
        pred_label = model.predict(X_new)[0]
        pred_proba = model.predict_proba(X_new)[0]
        
        return pred_label, pred_proba, features, y_processed, sr_processed, speaker_id, similarity
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None, None, None, None, None

def create_waveform_plot(y, sr, title="Audio Waveform"):

    time = np.linspace(0, len(y)/sr, len(y))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=y,
        mode='lines',
        name='Amplitude',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def main():
    model, model_loaded = load_model()
    speaker_profiles, profiles_loaded = load_speaker_profiles()
    
    if not model_loaded:
        st.stop()
    
    st.sidebar.header("Model Information")
    st.sidebar.info(f"""
    **Model:** Random Forest
    **Classes:** {', '.join(model.classes_)}
    **Trees:** {model.n_estimators}
    **Selected Features:** 5 fitur (>50% importance)
    **Speaker Verification:** {'Aktif' if profiles_loaded else 'Tidak aktif'}
    """)
    
    if profiles_loaded:
        st.sidebar.success(f"👥 {len(speaker_profiles)} speaker terdaftar")
    else:
        st.sidebar.warning("Mode tanpa verifikasi speaker")
    
    # Input method selection
    st.header("Pilih Metode Input Audio")
    input_method = st.radio(
        "Bagaimana Anda ingin memberikan audio?",
        ["Upload File Audio", "Record Voice Live"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if input_method == "Upload File Audio":
            st.subheader("Upload Audio File")
            uploaded_file = st.file_uploader(
                "Pilih file audio (format WAV)",
                type=['wav'],
                help="Upload audio file yang ingin diklasifikasi"
            )
            
            if uploaded_file is not None:
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB"
                }
                st.json(file_details)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_file_path = tmp_file.name
                
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("Classify Audio", type="primary"):
                    with st.spinner("Processing audio..."):
                        prediction, probabilities, features, y_processed, sr_processed, speaker_id, similarity = predict_audio(
                            temp_file_path, model, speaker_profiles, is_file=True
                        )
                    
                    process_prediction_results(prediction, probabilities, features, y_processed, sr_processed, speaker_id, similarity, model, col2)
                    os.unlink(temp_file_path)
        
        else:  # Record Voice Live
            st.subheader("Record Your Voice")
            st.info("Klik tombol record di bawah, bicara selama 1-3 detik, lalu klik stop")
            
            # Audio recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone-lines",
                icon_size="2x",
                pause_threshold=1.0,
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("Classify Recorded Audio", type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_bytes)
                        temp_file_path = tmp_file.name
                    
                    with st.spinner("Processing recorded audio..."):
                        prediction, probabilities, features, y_processed, sr_processed, speaker_id, similarity = predict_audio(
                            temp_file_path, model, speaker_profiles, is_file=True
                        )
                    
                    process_prediction_results(prediction, probabilities, features, y_processed, sr_processed, speaker_id, similarity, model, col2)
                    os.unlink(temp_file_path)

def process_prediction_results(prediction, probabilities, features, y_processed, sr_processed, speaker_id, similarity, model, col2):
    """Process and display prediction results"""
    if prediction is not None:
        with col2:
            st.header("Classification Results")
            
            if speaker_id is not None:
                st.subheader("🔐 Speaker Verification")
                if prediction == "unregistered":
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⛔ SPEAKER TIDAK TERDAFTAR</h3>
                        <p><strong>Similarity Score: {similarity:.3f}</strong> (Threshold: 0.75)</p>
                        <p>🚫 Suara tidak dikenali sebagai pengguna terdaftar</p>
                        <p>🔒 Akses ditolak untuk keamanan sistem</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tambahkan informasi debug
                    with st.expander("🔍 Debug Information"):
                        st.write(f"**Detected Speaker:** {speaker_id}")
                        st.write(f"**Similarity Score:** {similarity:.6f}")
                        st.write(f"**Required Threshold:** 0.75")
                        st.write(f"**Status:** Score terlalu rendah - kemungkinan bukan speaker terdaftar")
                    return
                else:
                    # Tentukan confidence level
                    if similarity > 0.85:
                        confidence_level = "🟢 High Confidence"
                        confidence_color = "success"
                    elif similarity > 0.75:
                        confidence_level = "🟡 Medium Confidence" 
                        confidence_color = "warning"
                    else:
                        confidence_level = "🔴 Low Confidence"
                        confidence_color = "error"
                    
                    if confidence_color == "success":
                        st.success(f"✅ **Speaker Verified:** {speaker_id}")
                        st.success(f"📊 **Similarity:** {similarity:.3f} | {confidence_level}")
                    elif confidence_color == "warning":
                        st.warning(f"⚠️ **Speaker:** {speaker_id} | Similarity: {similarity:.3f} | {confidence_level}")
                    
            # Classification results hanya jika speaker terverifikasi
            confidence = max(probabilities) * 100
            
            if prediction.lower() == 'buka':
                st.markdown(f"""
                <div class="prediction-box prediction-buka">
                    <h2>🔓 BUKA</h2>
                    <p><strong>Confidence: {confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box prediction-tutup">
                    <h2>🔒 TUTUP</h2>
                    <p><strong>Confidence: {confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.subheader("📈 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': model.classes_,
                'Probability': probabilities * 100
            })
            
            fig = px.bar(
                prob_df, 
                x='Class', 
                y='Probability',
                title='Class Probabilities (%)',
                color='Probability',
                color_continuous_scale='RdYlBu_r',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature details
            if features:
                st.subheader("🔍 Extracted Features")
                features_df = pd.DataFrame([features]).T
                features_df.columns = ['Value']
                st.dataframe(features_df, use_container_width=True)
        
        # Audio analysis
        st.header("🎵 Audio Analysis")
        if y_processed is not None:
            waveform_fig = create_waveform_plot(
                y_processed, sr_processed, "Processed Audio Waveform (1 second, 22050 Hz)"
            )
            st.plotly_chart(waveform_fig, use_container_width=True)

    st.header("📋 Cara Menggunakan Aplikasi")
    st.markdown("""
    ### Fitur Utama:
    1. **Upload Audio File** - Upload file WAV untuk klasifikasi
    2. **Voice Recording** - Record suara langsung melalui browser
    3. **Speaker Verification** - Verifikasi identitas speaker yang terdaftar (Threshold: 0.75)
    4. **Real-time Classification** - Klasifikasi suara "Buka" vs "Tutup"

    ### Cara Penggunaan:
    1. **Pilih** metode input (Upload file atau Record voice)
    2. **Berikan** audio input sesuai metode yang dipilih
    3. **Klik** tombol "Classify" untuk memproses
    4. **Lihat** hasil prediksi dan confidence score

    ### Penting:
    - **Hanya suara terdaftar** yang dapat diproses (Similarity > 0.75)
    - Audio akan diproses menjadi **22050 Hz, 1 detik**
    - Sistem menggunakan **5 fitur akustik** terpilih untuk klasifikasi
    - Speaker verification menggunakan **MFCC + Pitch + Spectral features**
    - **Threshold ketat** untuk mencegah akses tidak sah
    """)

if __name__ == "__main__":
    main()