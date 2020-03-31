#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 8              #
# 24 March 2020             #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Part 1

# This function is obviously pretty useless, but it does represent the
# simplified form of the expression for a_k which is asked for in the lab
def a_k(k):
    return 0

def b_k(k):
    return (2 / (k * np.pi)) * (1 - np.cos(k*np.pi))

a_0 = a_k(0)
a_1 = a_k(1)
b_1 = b_k(1)
b_2 = b_k(2)
b_3 = b_k(3)

print("a_0 = {}, a_1 = {}, b_1 = {}, b_2 = {}, b_3 = {}".format(a_0, a_1, b_1, b_2, b_3))

#%% Part 2

def fourier_series(N, T, t):
    
    omega_0 = (2*np.pi)/T
    x = np.zeros(t.shape)
    
    for i in range(1, N+1):  # had some trouble with this; python iterates to max_val-1 so, I had to add +1
        x += (b_k(i) * np.sin(i * omega_0 * t))
    
    return x

steps = 1e-3 # define step size
t_step = 20
t = np.arange(0, t_step + steps, steps)

x_1 = fourier_series(1, 8, t)
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_1)
plt.grid()
x_3 = fourier_series(3, 8, t)
plt.subplot(3, 1, 2)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_3)
plt.grid()
x_15 = fourier_series(15, 8, t)
plt.subplot(3, 1, 3)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_15)
plt.grid()
plt.suptitle('Fourier Series Approximations of x(t) (N=1, N=3, N=15)')

x_50 = fourier_series(50, 8, t)
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_50)
plt.grid()
x_150 = fourier_series(150, 8, t)
plt.subplot(3, 1, 2)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_150)
plt.grid()
x_1500 = fourier_series(1500, 8, t)
plt.subplot(3, 1, 3)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.plot(t, x_1500)
plt.grid()
plt.suptitle('Fourier Series Approximations of x(t) (N=50, N=150, N=1500)')