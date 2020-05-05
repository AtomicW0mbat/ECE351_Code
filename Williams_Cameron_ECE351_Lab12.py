#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 12             #
# 5 May 2020                #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import control as con
import pandas as pd

# fft_clean function from Lab 9
# had to change scipy.fftpack calls to scipy.fft since I updated to scipy v1.4.1
def fft_clean(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = scipy.fft.fft(x)  # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fft.fftshift(X_fft)  # shift zero frequency
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

def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0],x[-1],0,color='r')
    ax.vlines(x, 0, y, color=color, linestyle=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
# function I created for duplicating poles in a transfer function
# I used it for creating low-pass stage and high-pass stage, each multiples
# of transfer functions with the same pole
def stage_poles(num, den, poles):
    num_stage = num
    den_stage = den
    for i in range (poles-1):
        num_stage = sig.convolve(num_stage, num)
        den_stage = sig.convolve(den_stage, den)
    return num_stage, den_stage
    
# function I created for easily showing frequency response of the filter at a given frequency
def response(freq_hz, R, C1, C2, poles):
    w = 2*np.pi * freq_hz
    out = np.abs((1/(R*C1) / (1j*w+ 1/(R*C1)))**poles * ((1j*w)/(1j*w+ 1/(R*C2)))**poles)
    return 20*np.log10(out)

#load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

# Using a fourier transform to find frequency conetent of provided signal

fs = 1/1e-6 # sampling interval 1e-6 as seen in the provided .csv file
T = 1e-6
#t = np.arange(0, 0.05, T) # ending value 0.05 evident in CSV file and plot

sensor_freq, sensor_mag, sensor_phi = fft_clean(sensor_sig, fs)
fig , (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize =(10, 8))
plt.subplot(ax1) # entire frequency contents
make_stem(ax1, sensor_freq, sensor_mag)
plt.grid()
plt.subplot(ax2) # low frequency contents showing 
make_stem(ax2, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-100, 100)
plt.ylabel('Magnitude')
plt.subplot(ax3) # middle frequency contents showing sensor data
make_stem(ax3, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-10000, 10000)
plt.subplot(ax4) # high frequency contents
make_stem(ax4, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-75000, 75000)
plt.xlabel('f[Hz]')
plt.suptitle("Frequency content of sensor_sig")
plt.show()

# print relevant frequencies
for i in range(len(sensor_freq)):
    if (sensor_freq[i] > 0 and sensor_mag[i] > 0.4):
        print("Magnitude {} at frequency {}".format(sensor_mag[i], sensor_freq[i]))
print('\n')

steps = 10
omega = np.arange(50, 600000, steps)

f_c1 = 20000 # corner frequency for the low pass filter stages
f_c2 = 150 # corner frequency for the high pass filter stages
R = 10e3 # selected somewhat arbitrarily
C1 = 1 / (2*np.pi*f_c1*R)
C2 = 1 / (2*np.pi*f_c2*R)
poles = 4 # poles for each stage to have
# could make this more flexible by allowing different poles for low-pass
# and high-pass stages
print("Capacitor value for C1: {}".format(C1))
print("Capacitor value for C2: {}".format(C2))

# transfer function of a basic low pass pole
lps_num = [1/(R*C1)]
lps_den = [1, 1/(R*C1)]

# transfer function of a basic high pass pole
hps_num = [1, 0]
hps_den = [1, 1/(R*C2)]

lp_stage_num, lp_stage_den = stage_poles(lps_num, lps_den, poles)
hp_stage_num, hp_stage_den = stage_poles(hps_num, hps_den, poles)

#combined stages
num = sig.convolve(lp_stage_num, hp_stage_num)
den = sig.convolve(lp_stage_den, hp_stage_den)

Y_freq, Y_mag, Y_phase = sig.bode((num, den), w=omega, n=steps)

# Plot for frequency response
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.ylabel('|H(jω)| (dB)')
plt.semilogx(Y_freq, Y_mag)
plt.grid()
plt.suptitle('Bode plot of inital design')
plt.subplot (2, 1, 2)
plt.ylabel('/_H(jω) (degrees)')
plt.xlabel('ω')
plt.semilogx(Y_freq, Y_phase)
plt.grid()

sys = con.TransferFunction(num, den)
mag, phase, omega = con.bode(sys, omega, dB = True, Hz = True, deg = True, Plot = True)

# Evaluate transfer function at identified frequencies of importance
# Shows that desired attenuation has been achieved
important_freqs = [60, 1800, 1900, 2000, 49900, 50000, 51000, 100000]
print("Attenuation of important frequencies is as follows:")
for i in range(len(important_freqs)):
    print("{} Hz attenuated by {:.2f} dB".format(important_freqs[i], response(important_freqs[i], R, C1, C2, poles)))


# Filter noisy input to produce output
d_num, d_den = sig.bilinear(num, den, 600000)
# sample frequency must be high enough for highest frequency component in the incoming signal
y_out = sig.lfilter(d_num, d_den, sensor_sig)

plt.figure(figsize = (10, 7))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.plot(t, y_out)
plt.grid()
plt.title('Filtered Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

# Run output through fourier transform to show filtered signal contents
sensor_freq, sensor_mag, sensor_phi = fft_clean(y_out, fs)
fig , (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize =(10, 8))
plt.subplot(ax1) # entire frequency contents
make_stem(ax1, sensor_freq, sensor_mag)
plt.grid()
plt.subplot(ax2) # low frequency contents showing 
make_stem(ax2, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-100, 100)
plt.ylabel('Magnitude')
plt.subplot(ax3) # middle frequency contents showing sensor data
make_stem(ax3, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-10000, 10000)
plt.subplot(ax4) # high frequency contents
make_stem(ax4, sensor_freq, sensor_mag)
plt.grid()
plt.xlim(-75000, 75000)
plt.xlabel('f[Hz]')
plt.suptitle("Frequency content of sensor_sig")
plt.show()