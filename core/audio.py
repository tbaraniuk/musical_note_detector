import numpy as np


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr, n_fft, n_mels=40, fmin=0, fmax=None):
    fmax = fmax or sr / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    hz_points = mel_to_hz(mel_points)

    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, (n_fft // 2) + 1))

    for i in range(1, n_mels + 1):
        left = bin_indices[i - 1]
        center = bin_indices[i]
        right = bin_indices[i + 1]

        # build rising slope
        denom_left = center - left
        if denom_left > 0:
            for j in range(left, center):
                filterbank[i - 1, j] = (j - left) / denom_left

        # build falling slope
        denom_right = right - center
        if denom_right > 0:
            for j in range(center, right):
                filterbank[i - 1, j] = (right - j) / denom_right

    return filterbank

def generate_note(freq, duration=0.5, sr=44100, num_harmonics=6, noise_std=0.03):
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        amplitude = np.random.uniform(0.5, 1.5) / k
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * k * t + phase)
    
    envelope = np.exp(-3 * t / duration)
    signal *= envelope
    
    noise = np.random.normal(0, noise_std, len(signal))
    signal += noise
    
    return signal / np.max(np.abs(signal))

def generate_training_samples(freq, n_samples=1, noise_std=0.02, wobble=2.0):
    samples = []
    for _ in range(n_samples):
        pitch_wobble = freq + np.random.uniform(-wobble, wobble)
        signal = generate_note(pitch_wobble)
        samples.append(signal)
                
    return samples

def get_fft_spectrum(audio_signal, n_fft):
    N = len(audio_signal)
    n = np.arange(N)
    
    hann_window = 0.5 - 0.5 * np.cos((2 * np.pi * n) / (N - 1))
    
    windowed_signal = audio_signal * hann_window
    
    fft_complex = np.fft.rfft(windowed_signal, n=n_fft)
        
    magnitudes = np.abs(fft_complex)
    magnitudes = (magnitudes / N) * 2 
    
    return np.log(magnitudes + 1e-9)

def get_mel_spectrum(audio_signal, n_fft, filterbank):
    N = len(audio_signal)
    n = np.arange(N)
    
    hann_window = 0.5 - 0.5 * np.cos((2 * np.pi * n) / (N - 1))
    windowed_signal = audio_signal * hann_window
    
    fft_complex = np.fft.rfft(windowed_signal, n=n_fft)
    
    magnitudes = np.abs(fft_complex)
    magnitudes = (magnitudes / N) * 2
    
    mel_magnitudes = filterbank @ magnitudes
    
    return np.log(mel_magnitudes + 1e-9)
