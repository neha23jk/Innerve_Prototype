import numpy as np
import sounddevice as sd

# Config
FS = 44100       # Sample rate
DURATION = 0.1   # Seconds per bit (increase this if detection is bad)
FREQ_0 = 1000    # Hz for bit 0
FREQ_1 = 1200    # Hz for bit 1

def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(FS * duration), endpoint=False)
    # Generate sine wave
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def send_message(text):
    bits = text_to_bits(text)
    audio_signal = []
    
    # 1. Add Start Tone (Handshake) - 2000Hz for 0.5s
    audio_signal.append(generate_tone(2000, 0.5))
    
    # 2. Add Data Tones
    for bit in bits:
        if bit == '0':
            audio_signal.append(generate_tone(FREQ_0, DURATION))
        else:
            audio_signal.append(generate_tone(FREQ_1, DURATION))
            
    # 3. Concatenate
    full_signal = np.concatenate(audio_signal)
    
    # 4. Add Noise (Spectral Masking)
    noise = np.random.normal(0, 0.1, full_signal.shape) # Adjust 0.1 for volume
    masked_signal = full_signal + noise
    
    # Play
    sd.play(masked_signal, FS)
    sd.wait()

# Usage
# send_message("SOS")