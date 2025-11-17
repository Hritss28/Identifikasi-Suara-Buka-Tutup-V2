import os
import librosa
import numpy as np
import pickle
from pathlib import Path

def extract_enhanced_speaker_features(y, sr=22050):
    
    # 1. MFCC features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # 2. Pitch/Fundamental frequency
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # 3. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    
    # 4. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Gabungkan semua features
    speaker_features = np.concatenate([
        mfcc_mean,           # 13 features
        mfcc_std,            # 13 features  
        [pitch_mean],        # 1 feature
        [np.mean(spectral_centroids)],  # 1 feature
        [np.mean(spectral_rolloff)],    # 1 feature
        [np.mean(zero_crossing)],       # 1 feature
        chroma_mean          # 12 features
    ])
    
    return speaker_features

def preprocess_audio_for_speaker_v2(y, sr, target_sr=22050):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Durasi 2-3 detik untuk speaker profiling (lebih panjang = lebih akurat)
    max_len = int(sr * 2.0)  # 2 detik
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        # Ambil bagian tengah jika terlalu panjang
        start = (len(y) - max_len) // 2
        y = y[start:start + max_len]
    
    # Normalisasi yang lebih halus
    y = y / (np.max(np.abs(y)) + 1e-6)
    
    # Noise reduction yang lebih konservatif
    y[np.abs(y) < 0.005] = 0.0
    
    return y, sr

def create_speaker_profiles_v2():    
    speaker_samples_dir = "speaker_samples"
    
    if not os.path.exists(speaker_samples_dir):
        print(f"Folder '{speaker_samples_dir}' tidak ditemukan!")
        return
    
    speaker_profiles = {}
    
    for speaker_name in os.listdir(speaker_samples_dir):
        speaker_dir = os.path.join(speaker_samples_dir, speaker_name)
        
        if not os.path.isdir(speaker_dir):
            continue
            
        print(f"\n🎤 Processing speaker: {speaker_name}")
        
        speaker_features_list = []
        
        # Process semua file audio untuk speaker ini
        for audio_file in os.listdir(speaker_dir):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(speaker_dir, audio_file)
                print(f"  📁 Processing: {audio_file}")
                
                try:
                    # Load audio
                    y, sr = librosa.load(audio_path, sr=22050)
                    
                    # Preprocess dengan algoritma yang diperbaiki
                    y_processed, sr_processed = preprocess_audio_for_speaker_v2(y, sr)
                    
                    # Extract features dengan algoritma enhanced
                    features = extract_enhanced_speaker_features(y_processed, sr_processed)
                    
                    # Normalize features
                    features_normalized = features / (np.linalg.norm(features) + 1e-8)
                    
                    speaker_features_list.append(features_normalized)
                    print(f"    Features extracted: {len(features)} dimensions")
                    
                except Exception as e:
                    print(f"    Error processing {audio_file}: {e}")
        
        if speaker_features_list:
            # Rata-rata dari semua samples untuk speaker ini
            avg_features = np.mean(speaker_features_list, axis=0)
            
            # Normalize lagi
            avg_features_normalized = avg_features / (np.linalg.norm(avg_features) + 1e-8)
            
            # Hitung variabilitas untuk quality check
            feature_std = np.std(speaker_features_list, axis=0)
            variability = np.mean(feature_std)
            
            speaker_profiles[speaker_name] = avg_features_normalized
            print(f"  Profile created for {speaker_name}")
            print(f"     - Samples used: {len(speaker_features_list)}")
            print(f"     - Feature dimensions: {len(avg_features_normalized)}")
            print(f"     - Variability score: {variability:.4f}")
        else:
            print(f"  No valid audio files found for {speaker_name}")
    
    # Save profiles
    if speaker_profiles:
        with open('speaker_profiles.pkl', 'wb') as f:
            pickle.dump(speaker_profiles, f)
        
        print(f"\nSpeaker profiles saved successfully!")
        print(f"Registered speakers: {list(speaker_profiles.keys())}")
        print("File saved as: speaker_profiles.pkl")
        
        # Test similarity antar speaker untuk quality check
        print(f"\nQuality Check - Inter-speaker similarity:")
        speakers = list(speaker_profiles.keys())
        if len(speakers) == 2:
            profile1 = speaker_profiles[speakers[0]]
            profile2 = speaker_profiles[speakers[1]]
            similarity = np.dot(profile1, profile2)
            print(f"   Similarity between {speakers[0]} and {speakers[1]}: {similarity:.4f}")
            if similarity > 0.9:
                print("   WARNING: Speakers too similar! Consider recording more diverse samples.")
            else:
                print("   Good separation between speakers.")
    else:
        print("\nNo speaker profiles created!")
if __name__ == "__main__":
    create_speaker_profiles_v2()