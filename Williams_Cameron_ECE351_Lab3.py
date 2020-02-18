#############################
# Cameron Williams          #
# ECE 351-51                #
# Lab Number 3              #
# 11 February 2020          #
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
    y = step_func(t-2) - step_func(t-9)
    return y
                
def func2(t):  # implemenation of function described in handout
    y = np.exp(-t)#*step_func(t)
    return y

def func3(t):  # implemenation of function described in handout
    y = ramp_func(t-2) * (step_func(t-2) - step_func(t-3)) +\
        ramp_func(4-t) * (step_func(t-3) - step_func(t-4))
    return y
     
steps = 1e-3 # define step size
t_step = 20
t = np.arange(0, t_step + steps, steps)
y1 = func1(t)
y2 = func2(t)
y3 = func3(t)

xmarks = np.arange(0, 21, 2)    # set x axis marks
ymarks = np.arange(0, 2, 0.5)   # set y axis marks
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)            # 3 rows, 1 column, 1st subplot
plt.ylabel('y_1(t)')
plt.plot(t, y1)
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, y2)
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('y_2(t)')
plt.subplot(3,1,3)
plt.plot(t,y3)
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('y_3(t)')
plt.xlabel('t')
plt.suptitle('RLC Circuit Signals')
plt.show()

#%% Part 2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Task 1: Convolution function
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
t = np.arange(0, 20 + steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN - 1], steps)

y1 = func1(t)
y2 = func2(t)
y3 = func3(t)

# Task 2: Convolve y1 and y2

conv12 = my_convolve(y1, y2) * steps
conv12check = sig.convolve(y1, y2)*steps

'''
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tExtended, conv12, label = 'User-Defined Convolution')
plt.plot(tExtended, conv12check, '--', label ='Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_1(t) * f_2(t)')
plt.title('Convolution of f_1 and f_2')
'''

# Task 3
conv23 = my_convolve(y2, y3) * steps
conv23check = sig.convolve(y2, y3) * steps

'''
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 2)
plt.plot(tExtended, conv23, label='User-defined')
plt.plot(tExtended, conv23check, label ='Built-in')
'''

# Task 4
conv13 = my_convolve(y1, y3) * steps
conv13check = sig.convolve(y1, y3) * steps

'''
plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 3)
plt.plot(tExtended, conv13, label='User-defined')
plt.plot(tExtended, conv13check, label ='Built-in')
plt.show()
'''

xmarks = np.arange(0, 41, 5)    # set x axis marks
ymarks = np.arange(0, 2, 0.5)   # set y axis marks
plt.figure(figsize = (10, 7))
plt.subplot(3, 2, 1)  # 6 plots total
plt.plot(tExtended, conv12, label='User-defined')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('y_1(t) * y_2(t)')
plt.subplot(3, 2, 2)
plt.plot(tExtended, conv12check, label='User-defined')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3, 2, 3)
plt.plot(tExtended, conv23, label='User-defined')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.ylabel('y_2(t) * y_3(t)')
plt.subplot(3, 2, 4)
plt.plot(tExtended, conv23check, label ='Built-in')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.subplot(3, 2, 5)
plt.plot(tExtended, conv13, label='User-defined')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.xlabel('t')
plt.ylabel('y_1(t) * y_3(t)')
plt.subplot(3, 2, 6)
plt.plot(tExtended, conv13check, label ='Built-in')
plt.xticks(xmarks)
plt.yticks(ymarks)
plt.grid()
plt.xlabel('t')
plt.suptitle('Convolutions: User-defined and Built-in')
plt.show()