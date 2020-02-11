#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 2              #
# 4 February 2020           #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy
import control
import pandas

#%% Part 1
def func1(t): # the only variable sent to the function is t
    y = np.zeros(t.shape) # initialize y(t) as an array of zeros
    
    for i in range(len(t)): #run the loop once for each index of t
        y[i] = np.cos(t[i])
    return y #send back the output stored in an array


steps = 1e-2 # define step size
t = np.arange(0, 10+ steps, steps)
y = func1(t) # call the function we just created

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) (w/ Good Resolution)')
plt.title('P1T2: y = cos(t)')

#%% Part 2

def step_func(t):
    y = np.zeros((len(t), 1))
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y
t = np.arange(-5, 5+ steps, steps)
y = step_func(t) # call the function we just created

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('u(t)')
plt.title('P2T2: Step Function')

def ramp_func(t):
    y = np.zeros((len(t), 1))
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y
t = np.arange(-5, 5+ steps, steps)
y = ramp_func(t) # call the function we just created

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('r(t)')
plt.title('P2T2: Ramp Function')


# r(t) + 5u(t-3) - r(t-3) - 2u(t-6) - 2r(t-6)

def func2(t):
    return ramp_func(t) + 5*step_func(t-3) - ramp_func(t-3)\
                - 2*step_func(t-6) - 2*ramp_func(t-6)
t = np.arange(-5, 10+ steps, steps)
y = func2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('P2T3: Figure 2 Plotted')

#%% Part 3
t = np.arange(-10, 5+ steps, steps)
y = func2(-t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(-t)')
plt.title('P3T1: Figure 2 Time Reversal')


t = np.arange(-1, 14+ steps, steps)
y = func2(t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t-4)')
plt.title('P3T2: Figure 2 Shifted 4')

t = np.arange(-15, 1+ steps, steps)
y = func2(-t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(-t-4)')
plt.title('P3T2: Figure 2 Reversed and Shifted 4')

t = np.arange(-2, 20+ steps, steps)
y = func2(t/2)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t/2)')
plt.title('P3T3: Figure 2 Halved Timescale')

t = np.arange(-1, 5+ steps, steps)
y = func2(2*t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(2t)')
plt.title('P3T3: Figure 2 Doubled Timescale')


# derivative
steps = 1e-3
t = np.arange(-1, 10+steps, steps)
y = ramp_func(t) + 5*step_func(t-3) - ramp_func(t-3)\
    - 2*step_func(t-6) - 2*ramp_func(t-6)
dt = np.diff(t)
dy = np.diff(y, axis=0)/dt
plt.figure(figsize=(10,7))
plt.plot(t, y, '--', label='y(t)')
plt.plot(t[range(len(dy))], dy[:,0], label='dy(t)dt')
plt.title('P3T3: Derivative WRT time')
plt.legend()
plt.grid()
plt.ylim([-2,10])
plt.show()