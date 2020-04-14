#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 10             #
# 7 April 2020              #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con # I didn't have this imported; had to fix it

#%% Part 1

def H_out (omega, R, L, C):
    mag = (omega/(R*C)) / (np.sqrt(omega**4 + [(1/(R*C))**2 - (2/(L*C))]*omega**2 + 1/(L*C)**2))
    mag_dB = 20*np.log10(mag)
    phase_rad = (np.pi/2) - np.arctan(omega/(R*C) / (-omega**2 + 1/(L*C)))
    phase_deg = phase_rad*180/np.pi 
    # my solution for fixing Plot 1 is pretty hacky and this wouldn't apply for all plots,
    # but this works for u
    for i in range(len(phase_deg)):
        if phase_deg[i] >= 90:
            phase_deg[i] -= 180
        else:
            continue
    
    return mag_dB, phase_deg

R = 1e3
L = 27e-3
C = 100e-9

steps = 10
omega = np.arange(1e3, 1e7, steps)

y1_mag, y1_phase = H_out(omega, R, L, C)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.ylabel('|H(jω)| (dB)')
plt.semilogx(omega, y1_mag)
plt.grid()
plt.suptitle('Task 1 - Bode Plot of H(s) from hand-derived Mag. and Phase')
plt.subplot (2, 1, 2)
plt.ylabel('/_H(jω) (degrees)')
plt.xlabel('ω')
plt.semilogx(omega, y1_phase)
plt.grid()

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

y2_freq, y2_mag, y2_phase = sig.bode((num, den), w=omega, n=steps)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.ylabel('|H(jω)| (dB)')
plt.semilogx(y2_freq, y2_mag)
plt.grid()
plt.suptitle('Task 2 - Bode Plot of H(s) via scipy.signal.bode()')
plt.subplot (2, 1, 2)
plt.ylabel('/_H(jω) (degrees)')
plt.xlabel('ω')
plt.semilogx(y2_freq, y2_phase)
plt.grid()

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.ylabel('|H(jω)| (dB)')
plt.semilogx(y2_freq, y2_mag)
plt.grid()
plt.suptitle('Task 2 - Bode Plot of H(s) via scipy.signal.bode()')
plt.subplot (2, 1, 2)
plt.ylabel('/_H(jω) (degrees)')
plt.xlabel('ω')
plt.semilogx(y2_freq, y2_phase)
plt.grid()

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega, dB = True, Hz = True, deg = True, Plot = True)
# use _ = ... to supress the output

#%% Part 2

fs = 100000 # sampling frequency must be at least two times that of the highest frequency
steps = 1/fs
t = np.arange(0, 0.01+steps, steps)
x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize = (10, 7))
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x)
plt.grid()
plt.suptitle('Plot of x(t)')

d_num, d_den = sig.bilinear(num, den, 200000)
y_out = sig.lfilter(d_num, d_den, x)

plt.figure(figsize = (10, 7))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.plot(t, y_out)
plt.grid()
plt.suptitle('Plot of y(t)')