#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 5              #
# 18 February 2020          #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def step_func(t):  # imported from previous lab; slightly modified
    y = np.zeros(t.shape)  # this was incorrect on my previous lab; fixed
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def ramp_func(t):  # imported from previous lab; slightly modified
    y = np.zeros(t.shape)  # this was incorrect on my previous lab; fixed
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

#%% Part 1
'''
# Example code for H(s) = (s + 2)/(s^2 + 3s + 8)

t = np.arange(0, 1.2e-3+steps, steps)

num = [0, 1, 2]
den = [1, 3, 8]

tout, yout = sig impulse((num, den), T = t)
'''
def sine_method(R, L, C, t):
    y = np.zeros(t.shape)
    
    alpha = -1/(2*R*C)
    omega = (1/2)*np.sqrt((1/(R*C))**2-4*(1/(np.sqrt(L*C)))**2 + 0*1j)
    p = alpha+omega
    g = 1/(R*C)*p
    g_mag = np.abs(g)
    g_angle = np.angle(g)  # angle in radians; most functions use this
    g_deg = np.rad2deg(g_angle)  # some functions take angle in degrees
    
    y = g_mag/np.abs(omega) * np.exp(alpha*t) * np.sin(np.abs(omega)*t + g_angle)*step_func(t)
    return y

steps = 1e-6 # define step size
t_step = 1.2e-3
t = np.arange(0, t_step + steps, steps)  # from 0 to 1.2 ms

R = 1e3
L = 27e-3
C = 100e-9

h1 = sine_method(R, L, C, t)  # using values from prelab circuit
num = [0, 1/(R*C),0]
den = [1, 1/(R*C), 1/(L*C)]

tout, yout = sig.impulse((num, den), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)            # 3 rows, 1 column, 1st subplot
plt.ylabel('calculated')
plt.ticklabel_format(style='scientific', scilimits=(0, 1))
plt.plot(t, h1)
plt.grid()
plt.subplot(2, 1, 2)
plt.ylabel('scipy.signal.impulse')
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t')
plt.ticklabel_format(style='scientific', scilimits=(0, 1))
plt.suptitle('Impulse Response h(t)')

#%% Part 2

tout, yout = sig.step((num, den), T = t)
plt.figure(figsize = (10, 7))
plt.ylabel('calculated')
plt.xlabel('t')
plt.ticklabel_format(style='scientific', scilimits=(0, 1))
plt.plot(t, h1)
plt.grid()
plt.suptitle('Step Response')