import numpy as np
import math
import matplotlib.pyplot as plt

N = 100
t = np.linspace(0,1,N)
f = np.array(([5], [6]))
y = np.sin(2*np.pi*f*t)

# Compute the FFT
fft_output = np.fft.fft(y, axis=1)

# Compute the frequencies
sampling_freq = 1 / (t[1] - t[0])  # Sampling frequency
freqs = np.fft.fftfreq(N, 1 / sampling_freq)

# Plot the FFT output
plt.figure(figsize=(8, 6))
plt.plot(freqs, np.abs(fft_output[0]))
plt.plot(freqs, np.abs(fft_output[1]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Sine Wave')
plt.grid(True)
plt.show()