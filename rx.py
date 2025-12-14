import sounddevice as sd
import numpy as np
import scipy.fftpack

# --- CONFIGURATION (Must match tx.py exactly) ---
FS = 44100          # Sample rate
BIT_DURATION = 0.1  # Seconds per bit
FREQ_0 = 1000       # Hz for '0'
FREQ_1 = 1200       # Hz for '1'
START_FREQ = 2000   # Hz for handshake
HANDSHAKE_TOLERANCE = 50 # Allow +/- 50Hz deviation

def get_dominant_freq(audio_chunk):
    """Performs FFT to find the loudest frequency in the chunk."""
    N = len(audio_chunk)
    # Apply Windowing (reduces spectral leakage)
    windowed_chunk = audio_chunk * np.hanning(N)
    
    # Fast Fourier Transform
    yf = scipy.fftpack.fft(windowed_chunk)
    xf = np.linspace(0.0, FS/2.0, N//2)
    
    # Get magnitude of positive frequencies
    amplitudes = 2.0/N * np.abs(yf[:N//2])
    
    # Find index of max amplitude
    max_idx = np.argmax(amplitudes)
    return xf[max_idx]

def decode_bits_to_text(bit_string):
    """Converts a string of 1s and 0s back to text."""
    # Split into 8-bit chunks
    chars = []
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        if len(byte) < 8: break # Ignore trailing bits
        try:
            chars.append(chr(int(byte, 2)))
        except ValueError:
            chars.append('?') # Error placeholder
    return "".join(chars)

def listen_and_decode():
    print("🎤 LISTENING... (Waiting for 2000Hz Handshake)")
    
    # 1. WAIT FOR HANDSHAKE
    # We record in small chunks (0.1s) checking for the start tone
    block_size = int(FS * 0.1) 
    
    while True:
        # Record a short chunk
        recording = sd.rec(block_size, samplerate=FS, channels=1, dtype='float64')
        sd.wait()
        
        # Check frequency
        dom_freq = get_dominant_freq(recording.flatten())
        
        # If we hear the 2000Hz tone (approx), start the main recording
        if abs(dom_freq - START_FREQ) < HANDSHAKE_TOLERANCE:
            print(f"✅ HANDSHAKE DETECTED! ({int(dom_freq)}Hz)")
            break

    # 2. RECORD THE MESSAGE
    # For this prototype, we will record a fixed 5 seconds to catch the message.
    # In a full version, you would encode the length in the header.
    print("🔴 RECORDING MESSAGE STREAM...")
    RECORD_SECONDS = 5 
    full_recording = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=1, dtype='float64')
    sd.wait()
    print("✅ RECORDING COMPLETE. PROCESSING...")

    # 3. DEMODULATE (FSK DECODING)
    # We slice the recording into chunks of BIT_DURATION
    audio_data = full_recording.flatten()
    samples_per_bit = int(FS * BIT_DURATION)
    total_bits = len(audio_data) // samples_per_bit
    
    detected_bits = ""
    
    for i in range(total_bits):
        # Extract the chunk for this bit
        start = i * samples_per_bit
        end = start + samples_per_bit
        chunk = audio_data[start:end]
        
        # Analyze frequencies in this specific chunk
        # We check the energy specifically at 1000Hz vs 1200Hz
        N = len(chunk)
        yf = scipy.fftpack.fft(chunk * np.hanning(N))
        freqs = scipy.fftpack.fftfreq(N, 1/FS)
        
        # Find indices for our target frequencies
        idx_0 = int(np.abs(freqs - FREQ_0).argmin())
        idx_1 = int(np.abs(freqs - FREQ_1).argmin())
        
        # Compare magnitudes
        mag_0 = np.abs(yf[idx_0])
        mag_1 = np.abs(yf[idx_1])
        
        if mag_1 > mag_0:
            detected_bits += "1"
        else:
            detected_bits += "0"

    print(f"raw bits: {detected_bits[:50]}...") # Print start of bits for debug
    
    # 4. CONVERT TO TEXT
    decoded_message = decode_bits_to_text(detected_bits)
    
    print("\n" + "="*30)
    print("📩 DECODED MESSAGE:")
    print(decoded_message)
    print("="*30)

if __name__ == "__main__":
    try:
        listen_and_decode()
    except KeyboardInterrupt:
        print("\nStopped.")