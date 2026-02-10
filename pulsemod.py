# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:22:03 2020

@author: cluff, jportillo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy import signal


def createPulsedSignal(carrier_f, pulse_f, stim_t=1000, duty_cycle=.5, dt=0.01,
                   phi1=0, phi2=np.pi, A1=1, A2=1, pre_t=100, post_t=100,
                   ramp_up_t=0, ramp_down_t=0, plot_on=1, h_on=1):

    """
        Units: [f] = Hz, [t] = ms
    """

    # convert to ms
    f1 = carrier_f/1000

    if duty_cycle == 0:
        duty_cycle = 0.001 # min duty cycle



    if carrier_f/1000 == pulse_f:
        f2 = pulse_f/1000
    else:
        f2 = (carrier_f+pulse_f)/1000

    stim_t = stim_t + ramp_up_t + ramp_down_t
    stim_tt = np.arange(0,stim_t+dt,dt)

    if f1 == 0:
        A1=0
        A2 = 2*A2

    I1_stim = A1*signal.square(2*np.pi*f1*stim_tt,duty_cycle) + A2*signal.square(2*np.pi*f2*stim_tt,duty_cycle)

    if ramp_up_t:
        ramp_up_size = round(ramp_up_t/dt)
        ramp_vec = np.arange(0,1,1/ramp_up_size)

        I1_stim[0:ramp_vec.size] = ramp_vec*I1_stim[0:ramp_vec.size]


    if ramp_down_t:
        ramp_down_size = round(ramp_down_t/dt)
        ramp_vec = np.arange(0,1,1/ramp_down_size)[::-1]

        I1_stim[-ramp_vec.size:] = ramp_vec*I1_stim[-ramp_vec.size:]


    if pre_t or post_t:
        pre_zeros = np.zeros([1,int(pre_t/dt)])[0]
        post_zeros = np.zeros([1,int(post_t/dt)])[0]

        I1_stim = np.concatenate([pre_zeros,I1_stim,post_zeros])
        tot_t = np.arange(0,stim_t+post_t+pre_t+dt,dt)


    if plot_on:
        plt.figure()
        plt.plot(tot_t,I1_stim)
        plt.xlim([0,tot_t[-1]])

        if h_on:
            y_h = hilbert(I1_stim)
            y_env = np.abs(y_h)
            plt.plot(tot_t, y_env, 'r')

    return I1_stim, tot_t




if __name__ == '__main__':
    carrier_f = 0
    pulse_f = 0.01
    dc = .5
    A=50/2
    I, t_tot = createPulsedSignal(carrier_f, pulse_f, stim_t=3000, duty_cycle=dc, dt=0.01,
                   phi1=0, phi2=np.pi, A1=A, A2=A, pre_t=50, post_t=50,
                   ramp_up_t=500, ramp_down_t=500, plot_on=1, h_on=0)
