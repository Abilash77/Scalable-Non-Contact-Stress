import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_audio_features(audio_path=None, audio_segment=None, sr=16000, n_mfcc=13):
    """
    Extracts paper-specified voice biomarkers from a single 16kHz audio window.
    Features: MFCCs, Chroma, Mel, Spectral Contrast, ZCR, RMS (Loudness),
    Pitch (F0), Jitter, Shimmer, Formants, Silence Ratio.
    """
    try:
        if audio_segment is None and audio_path is not None:
            audio_segment, sr = librosa.load(audio_path, sr=sr)
            
        if audio_segment is None or len(audio_segment) == 0:
            return np.zeros(169)
            
        # 1. Single STFT for efficiency
        stft = np.abs(librosa.stft(audio_segment))
        
        # 2. Base Features from STFT
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0) # 12
        
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0) # 7
        
        # 3. Base Features from raw audio
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0) # 13
        
        mel = librosa.feature.melspectrogram(y=audio_segment, sr=sr)
        mel_mean = np.mean(mel.T, axis=0) # 128
        
        zcr = librosa.feature.zero_crossing_rate(audio_segment)
        zcr_mean = np.array([np.mean(zcr)]) # 1
        
        # Loudness (RMS)
        rms = librosa.feature.rms(y=audio_segment)
        rms_mean = np.array([np.mean(rms)]) # 1
        rms_mean_float = float(np.mean(rms))
        
        # 4. Pitch, Jitter, Shimmer
        f0 = librosa.yin(audio_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0 # 1
        
        # Jitter: cycle-to-cycle pitch perturbation (approximation)
        if len(f0_valid) > 1:
            jitter = float(np.mean(np.abs(np.diff(f0_valid))) / (f0_mean + 1e-6))
        else:
            jitter = 0.0 # 1
            
        # Shimmer: cycle-to-cycle amplitude perturbation (approximation from RMS)
        rms_flat = rms.flatten()
        if len(rms_flat) > 1:
            shimmer = float(np.mean(np.abs(np.diff(rms_flat))) / (rms_mean_float + 1e-6))
        else:
            shimmer = 0.0 # 1
            
        # 5. Formants
        a = librosa.lpc(audio_segment, order=8)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted(angz * (sr / (2 * np.pi)))
        f1 = freqs[1] if len(freqs) > 1 else 0.0
        f2 = freqs[2] if len(freqs) > 2 else 0.0
        f3 = freqs[3] if len(freqs) > 3 else 0.0
        formants = np.array([f1, f2, f3]) # 3
        
        # 6. Pause-related features (Silence Ratio)
        # Assuming silence is RMS < threshold
        threshold = 0.01
        silence_ratio = np.sum(rms_flat < threshold) / len(rms_flat) if len(rms_flat) > 0 else 0.0 # 1
        
        # Combine all features
        features = np.hstack([
            mfccs_mean,          # 13
            chroma_mean,         # 12
            mel_mean,            # 128
            contrast_mean,       # 7
            zcr_mean,            # 1
            rms_mean,            # 1
            np.array([f0_mean, jitter, shimmer]), # 3
            formants,            # 3
            np.array([silence_ratio]) # 1
        ]) # Total: 13+12+128+7+1+1+3+3+1 = 169 features
        
        return features
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return np.zeros(169)
