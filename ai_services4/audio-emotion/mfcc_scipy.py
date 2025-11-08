"""
Pure scipy/numpy MFCC extraction - No librosa/numba dependency
Works on Alpine Linux with system packages
"""
import numpy as np
import scipy.signal
import scipy.fft
from scipy.fftpack import dct
import soundfile as sf
from typing import Tuple


def hz_to_mel(hz):
    """Convert Hz to Mel scale"""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    """Convert Mel scale to Hz"""
    return 700 * (10**(mel / 2595) - 1)


def create_mel_filterbank(n_fft, n_mels, sample_rate, fmin=0, fmax=None):
    """
    Create Mel filterbank matrix
    
    Args:
        n_fft: FFT size
        n_mels: Number of Mel bands
        sample_rate: Sample rate of audio
        fmin: Minimum frequency
        fmax: Maximum frequency (defaults to sample_rate/2)
    
    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = sample_rate / 2
    
    # Convert Hz to Mel
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    
    # Create n_mels+2 points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz points to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # Create filterbank
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins))
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # Rising slope
        for j in range(left, center):
            filterbank[i, j] = (j - left) / (center - left)
        
        # Falling slope
        for j in range(center, right):
            filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


def extract_mfcc(audio_path: str, n_mfcc: int = 13, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128, 
                 sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Extract MFCC features from audio file using scipy
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of Mel bands
        sample_rate: Target sample rate
    
    Returns:
        Tuple of (mfcc_features, actual_sample_rate)
    """
    # Load audio file
    audio, sr = sf.read(audio_path, dtype='float32')
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed (simple linear interpolation)
    if sr != sample_rate:
        duration = len(audio) / sr
        new_length = int(duration * sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        )
        sr = sample_rate
    
    # Pre-emphasis filter
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Frame the signal
    frame_length = n_fft
    frames = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frames.append(audio[i:i + frame_length])
    
    if len(frames) == 0:
        # Audio too short, pad it
        frames = [np.pad(audio, (0, frame_length - len(audio)))]
    
    frames = np.array(frames)
    
    # Apply Hamming window
    window = np.hamming(frame_length)
    frames = frames * window
    
    # Compute STFT magnitude
    magnitude = np.abs(scipy.fft.rfft(frames, n=n_fft))
    
    # Create Mel filterbank
    mel_filterbank = create_mel_filterbank(n_fft, n_mels, sr)
    
    # Apply Mel filterbank
    mel_spectrum = np.dot(magnitude, mel_filterbank.T)
    
    # Convert to log scale
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
    log_mel_spectrum = np.log(mel_spectrum)
    
    # Apply DCT to get MFCC
    mfcc = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    
    return mfcc.T, sr


def extract_mfcc_statistics(mfcc: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from MFCC
    
    Args:
        mfcc: MFCC features of shape (n_mfcc, n_frames)
    
    Returns:
        Feature vector with mean, std, min, max for each coefficient
    """
    features = []
    
    # For each MFCC coefficient
    for coef in mfcc:
        features.extend([
            np.mean(coef),
            np.std(coef),
            np.min(coef),
            np.max(coef)
        ])
    
    return np.array(features)
