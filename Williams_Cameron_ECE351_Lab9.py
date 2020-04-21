#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 9              #
# 31 March 2020             #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

def fft(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = scipy.fftpack.fft(x)  # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)  # shift zero frequency
                                                   # componenets to the center
                                                   # of the spectrum
    freq = np.arange(-N/2, N/2) * fs/N  # compute the frequencies for the output
                                        # signal, (fs is the sampling frequency
                                        # and needs to be defined previously in
                                        # your code)
    X_mag = np.abs(X_fft_shifted) / N  # compute the magnitude of the signal
    X_phi = np.angle(X_fft_shifted)  # compute the phases of the signal
    
    return freq, X_mag, X_phi

def fft_clean(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = scipy.fftpack.fft(x)  # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)  # shift zero frequency
                                                   # componenets to the center
                                                   # of the spectrum
    freq = np.arange(-N/2, N/2) * fs/N  # compute the frequencies for the output
                                        # signal, (fs is the sampling frequency
                                        # and needs to be defined previously in
                                        # your code)
    X_mag = np.abs(X_fft_shifted) / N  # compute the magnitude of the signal
    X_phi = np.angle(X_fft_shifted)  # compute the phases of the signal
    for i in range(len(X_phi)):
        if (np.abs(X_mag[i]) < 1e-10):
            X_phi[i] = 0
        else:
            continue
    
    return freq, X_mag, X_phi

# imported from lab 8

def a_k(k):
    return 0

def b_k(k):
    return (2 / (k * np.pi)) * (1 - np.cos(k*np.pi))    

def fourier_series(N, T, t):
    
    omega_0 = (2*np.pi)/T
    x = np.zeros(t.shape)
    
    for i in range(1, N+1):  # had some trouble with this; python iterates to max_val-1 so, I had to add +1
        x += (b_k(i) * np.sin(i * omega_0 * t))
    
    return x

#%% Part 1

fs = 10000
T = 1/fs
t = np.arange(0, 2, T)

y1 = np.cos(2*np.pi*t)
y1_freq, y1_mag, y1_phi = fft(y1, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y1)
plt.grid()
plt.title('Task 1 - User-Defined FFT of x(t)=cos(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y1_freq, y1_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y1_freq, y1_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y1_freq, y1_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y1_freq, y1_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()


#%% Part 2

y2 = 5*np.sin(2*np.pi*t)
y2_freq, y2_mag, y2_phi = fft(y2, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y2)
plt.grid()
plt.title('Task 2 - User-Defined FFT of x(t)=5sin(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y2_freq, y2_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y2_freq, y2_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y2_freq, y2_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y2_freq, y2_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

#%% Part 3

y3 = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)**2
y3_freq, y3_mag, y3_phi = fft(y3, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y3)
plt.grid()
plt.title('Task 3 - User-Defined FFT of x(t)=2cos((2*pi*2*t)-2)+sin^2((2*pi*6*t)+3)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y3_freq, y3_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y3_freq, y3_mag)
plt.xlim(-15, 15)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y3_freq, y3_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y3_freq, y3_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

#%% Part 4

y1 = np.cos(2*np.pi*t)
y1_freq, y1_mag, y1_phi = fft_clean(y1, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y1)
plt.grid()
plt.title('Task 4 - User-Defined FFT (clean) of x(t)=cos(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y1_freq, y1_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y1_freq, y1_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y1_freq, y1_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y1_freq, y1_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

y2 = 5*np.sin(2*np.pi*t)
y2_freq, y2_mag, y2_phi = fft_clean(y2, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y2)
plt.grid()
plt.title('Task 4 - User-Defined FFT (clean) of x(t)=5sin(2*pi*t)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y2_freq, y2_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y2_freq, y2_mag)
plt.xlim(-2.0, 2.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y2_freq, y2_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y2_freq, y2_phi)
plt.xlim(-2.0, 2.0)
plt.ylim(-4, 4)
plt.grid()
plt.show()

y3 = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)**2
y3_freq, y3_mag, y3_phi = fft_clean(y3, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, y3)
plt.grid()
plt.title('Task 4 - User-Defined FFT (clean) of x(t)=2cos((2*pi*2*t)-2)+sin^2((2*pi*6*t)+3)')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(y3_freq, y3_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(y3_freq, y3_mag)
plt.xlim(-15, 15)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(y3_freq, y3_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(y3_freq, y3_phi)
plt.xlim(-2.0, 2.0)
plt.xlim(-15, 15)
plt.grid()
plt.show()

#%% Part 5

t = np.arange(0, 16, T)

x15 = fourier_series(15, 8, t)
x15_freq, x15_mag, x15_phi = fft_clean(x15, fs)
plt.figure(figsize = (15, 12))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.plot(t, x15)
plt.grid()
plt.title('Task 5 - User-Defined FFT (clean) of Lab 8 Fourier Series Approximation N=15')
plt.subplot(3, 2, 3)
plt.ylabel('|X(f)|')
plt.stem(x15_freq, x15_mag)
plt.grid()
plt.subplot(3, 2, 4)
plt.stem(x15_freq, x15_mag)
plt.xlim(-5.0, 5.0)
plt.grid()
plt.subplot(3, 2, 5)
plt.xlabel('f[Hz]')
plt.ylabel('\_X(f)')
plt.stem(x15_freq, x15_phi)
plt.ylim(-4, 4)
plt.grid()
plt.subplot(3, 2, 6)
plt.xlabel('f[Hz]')
plt.stem(x15_freq, x15_phi)
plt.xlim(-5.0, 5.0)
plt.ylim(-4, 4)
plt.grid()