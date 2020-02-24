#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 4              #
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
def func1(t):  # implemenation of function described in handout
    y = np.exp(2*t)*step_func(1-t)
    return y
                
def func2(t):  # implemenation of function described in handout
    y = step_func(t-2) - step_func(t-6)
    return y

def func3(t):  # implemenation of function described in handout
    y = np.zeros(t.shape)
    pi = np.pi
    omega = 2*pi*0.25  # omega = 2pif
    for i in range(len(t)): #run the loop once for each index of t
        y[i] = np.cos(omega*t[i])
    y = y * step_func(t)
    return y
     
steps = 1e-2 # define step size
t_step = 10
t = np.arange(-10, t_step + steps, steps)

h1 = func1(t)
h2 = func2(t)
h3 = func3(t)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)            # 3 rows, 1 column, 1st subplot
plt.ylabel('h_1(t)')
plt.plot(t, h1)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(-2, 10, 2)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, h2)
xmarks = np.arange(-10, 11, 2)    # set x axis marks
ymarks = np.arange(-0.5, 2.5, 0.5)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('h_2(t)')
plt.subplot(3,1,3)
plt.plot(t, h3)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(-1.5, 2, 0.5)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('h_3(t)')
plt.xlabel('t')
plt.suptitle('Transfer Functions')
plt.show()

#%% Part 2

def my_convolve(g, h):
    Ng = len(g)
    Nh = len(h)
    g_ext = np.append(g, np.zeros((1, Nh - 1)))  # double the length of g
    h_ext = np.append(h, np.zeros((1, Ng - 1)))  # double the length of h
    rslt = np.zeros(g_ext.shape)  # prepare the result array
    
    for i in range(Nh + Ng - 2):  # iterate through length of original h and g,
                                  # minus 2 to account for 0 index on each
        rslt[i] = 0  # initialize resulting sum
        for j in range(Ng):  # now iterate through Ng
            if ((i - j + 1)> 0):  # for values where t is positive
                try:
                    rslt[i] += g_ext[j]*h_ext[i - j + 1]
                    # sum up the product of the areas under each curve at a given time
                except:
                    print(i, j)  # print values where an error occured
    return rslt

steps = 1e-2
t = np.arange(-10, 10 + steps, steps)  # t stays the same, tExtended is double
NN = len(t)
tExtended = np.arange(2*t[0], 2*t[NN - 1] + steps, steps)
# Should always start from the first index of t, not necissarily 0
u = step_func(t)

step_resp1 = my_convolve(h1, u) * steps
step_resp2 = my_convolve(h2, u) * steps
step_resp3 = my_convolve(h3, u) * steps

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)            # 3 rows, 1 column, 1st subplot
plt.ylabel('h_1(t) * u(t)')
plt.plot(tExtended, step_resp1)
plt.xlim(-10, 10)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(0, 6, 1)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(tExtended, step_resp2)
plt.xlim(-10, 10)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(0, 6, 1)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('h_2(t) * u(t)')
plt.subplot(3,1,3)
plt.plot(tExtended, step_resp3)
plt.xlim(-10, 10)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(-1, 1.5, 0.5)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('h_3(t) * u(t)')
plt.xlabel('t')
plt.suptitle('Step Responses')
plt.show()

#Task 2

calc_resp1 = 0.5*np.exp(2*tExtended)*step_func(1-tExtended) + 0.5*np.exp(2)*step_func(tExtended-1)
calc_resp2 = ramp_func(tExtended-2) - ramp_func(tExtended-6)
omega = (2*np.pi*0.25)
calc_resp3 = (1/omega)*(np.sin(omega*tExtended))*step_func(tExtended)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)            # 3 rows, 1 column, 1st subplot
plt.ylabel('h_1(t) * u(t)')
plt.plot(tExtended, calc_resp1)
plt.xlim(-10, 10)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(0, 6, 1)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3,1,2)
plt.ylabel('h_2(t) * u(t)')
plt.plot(tExtended, calc_resp2)
plt.xlim(-10, 10)
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3,1,3)
plt.ylabel('h_3(t) * u(t)')
plt.plot(tExtended, calc_resp3)
plt.xlim(-10, 10)
xmarks = np.arange(-10, 12, 2)    # set x axis marks
ymarks = np.arange(-1, 1.5, 0.5)   # set y axis marks
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.xlabel('t')
plt.suptitle('Calculated Step Responses')
plt.show()