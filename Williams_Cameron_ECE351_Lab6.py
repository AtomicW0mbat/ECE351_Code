#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 6              #
# 3 March 2020              #
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

# Python's k is gain, Sullivan's k is residue
def cosine_method(residue, poles, t):
    y = np.zeros(t.shape)
    
    for i in range(len(residue)):
        alpha = np.real(poles[i])  # residue and pole indexes correspond?
        omega = np.imag(poles[i])
        k_mag = np.abs(residue[i])
        k_angle = np.angle(residue[i])
        #k_deg = np.rad2deg(k_angle)
        y += k_mag * np.exp(alpha*t) * np.cos(omega*t + k_angle)*step_func(t)
    
    return y

#%% Part 1

steps = 1e-3 # define step size
t_step = 2
t = np.arange(0, t_step + steps, steps)

y1 = ( (1/2) - ((1/2)*np.exp(-4*t)) + (np.exp(-6*t)) )*step_func(t)
# mistyped this before; took some debugging to find that second sign was
# incorrectly typed as a plus originally

plt.figure(figsize = (10, 7))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.plot(t, y1)
plt.grid()
plt.suptitle('Calculated Step Response')


num = [1, 6, 12]
den = [1, 10, 24]
tout, yout = sig.step((num, den), T = t)

plt.figure(figsize = (10, 7))
plt.ylabel('y(t)')
plt.xlabel('t')
plt.plot(tout, yout)
plt.grid()
plt.suptitle('Step Response via scipy.signal.step()')

num = [1, 6, 12]
den = [1, 10, 24, 0]
# 0 added on the end because of step response; it raises the degree of each
# term, making it so there are 0 terms of degree 0

r, p, k = sig.residue(num, den)
# r, p, and k returned from residue are the residue, poles, and gain

print("Part 1 Residue Results:\n r = {}\n p = {}\n k = {}\n ".format(r, p, k))

#%% Part 2

num2 = [25250]
den2 = [1, 18, 218, 2036, 9085, 25250, 0]
# This is the system described in Task 4.3; 0 added on the end because
# step response adds an extra 1/s term to the right hand side of the equation

r2, p2, k2 = sig.residue(num2, den2)

print("Part 2 Residue Results:\n r2 = {}\n p2 = {}\n k2 = {}\n".format(r2, p2, k2))

# plotting via cosine method; k in Sullivan's notes is residue
# see cosine_method() function defined at top
steps = 1e-3 # define step size
t2_step = 4.5
t2 = np.arange(0, t2_step + steps, steps)

y2 = cosine_method(r2, p2, t2)
plt.figure(figsize = (10, 7))
plt.ylabel('y_2(t)')
plt.xlabel('t')
plt.plot(t2, y2)
plt.grid()
plt.suptitle('Time-domain Response via Cosine Method')


num3 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250]
tout2, yout2 = sig.step((num3, den3), T = t2)

plt.figure(figsize = (10, 7))
plt.ylabel('y_2(t)')
plt.xlabel('t')
plt.plot(tout2, yout2)
plt.grid()
plt.suptitle('Time-domain Response via scipy.signal.step()')