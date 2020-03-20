#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 7              #
# 10 March 2020             #
# Instructor Philip Hagen   #
#############################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Part 1

G_num = [1, 9]
G_den = sig.convolve([1, -6, -16], [1, 4])

A_num = [1, 4]
A_den = [1, 4, 3]

B = [1, 26, 168]

G_zeroes, G_poles, G_gain = sig.tf2zpk(G_num, G_den)
print("G(s) has \n zeroes: {}\n poles: {}\n".format(G_zeroes, G_poles))

A_zeroes, A_poles, A_gain = sig.tf2zpk(A_num, A_den)
print("A(s) has \n zeroes: {}\n poles: {}\n".format(A_zeroes, A_poles))

B_zeroes = np.roots(B)
print("B(s) has \n zeroes: {}\n".format(B_zeroes))

ol_num = sig.convolve(A_num, G_num)
print("Open Loop Num = ", ol_num)
ol_den = sig.convolve(A_den, G_den)
print("Open Loop Den = ", ol_den)

ol_zeroes, ol_poles, ol_gain = sig.tf2zpk(ol_num, ol_den)
print("Open Loop has \n zeroes: {}\n poles: {}\n".format(ol_zeroes, ol_poles))

# If any poles are on the right side of the complex
# number plane, it's unstable. The real part of some of the poles 
# for the open loop transfer function are positive, so it's
# unstable.

steps = 1e-3 # define step size
t_step = 2
t = np.arange(0, t_step + steps, steps)

tout, yout = sig.step((ol_num, ol_den), T = t)

plt.figure(figsize = (10, 7))
plt.ylabel('h(t)')
plt.xlabel('t')
plt.plot(tout, yout)
plt.grid()
plt.suptitle('Step Response of the Open Loop')

# the plot supports my answer from Task 4 since the plotted 
# function "blows up"

cl_num = sig.convolve(A_num, G_num)
cl_den = sig.convolve((G_den + sig.convolve(B, G_num)), A_den)

cl_zeroes, cl_poles, cl_gain = sig.tf2zpk(cl_num, cl_den)
print("Closed Loop has \n zeroes: {}\n poles: {}\n".format(cl_zeroes, cl_poles))

# closed loop response is stable since there are no poles on 
# the right side of the complex number plane

tout2, yout2 = sig.step((cl_num, cl_den), T = t)
plt.figure(figsize = (10, 7))
plt.ylabel('h(t)')
plt.xlabel('t')
plt.plot(tout2, yout2)
plt.grid()
plt.suptitle('Step Response of the Closed Loop')

# The plot supports my answer since it converges
