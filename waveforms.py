#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy import signal


def createPulsedSignal(carrier_f, pulse_f, stim_t=1000, duty_cycle=1, dt=0.004,
                   phi1=0, phi2=np.pi, A1=1, A2=1, pre_t=100, post_t=100,
                   ramp_up_t=0, ramp_down_t=0, plot_on=1, waveform_mode=1, h_on=1):

    """
        Units: [f] = Hz, [t] = ms
    """
    # convert to ms
    carrier_f = carrier_f / 1000
    pulse_f = pulse_f / 1000

    if duty_cycle == 0:
        duty_cycle = 0.001  # min duty cycle

    f1 = carrier_f

    stim_t = stim_t + ramp_up_t + ramp_down_t  # ramps not in the stim time
    tot_t = np.arange(0, stim_t + dt, dt)


    #SINUSOIDAL TI
    if waveform_mode == 1:

        f_diff = pulse_f / duty_cycle  # f_diff not the same as pulse f!
        f2 = carrier_f + f_diff

        low_f = 0

        if pulse_f > 0: # TI case
            pulse_t = 1/f_diff

            break_t = 1/duty_cycle*pulse_t*(1-duty_cycle)
            cycle_t = pulse_t + break_t

            cycles = stim_t/cycle_t
            # allow not full cycles:
            half_cycles = int(np.ceil(cycles*2)) #int(round(cycles*2)) # number of pulses


            if carrier_f == 0: # low frequency pulsed signal case
                low_f = 1
                f1 = f2

                phi1 = -np.pi/2
                phi2 = -np.pi/2


        else: # high frequency case
            pulse_t = 0
            duty_cycle = 1 # allow only continuous high f

            phi1 = -np.pi/2
            phi2 = -np.pi/2


        # Creating pulses - indeces: 0-5000, 5001-10000, 10001-15000 etc.
        # - first cycle one timestep longer (0 time)


        ## Create signals I1 and I2
        if duty_cycle == 0.5:  # easier case - switch f2 to f1 during break

            I1 = A1*np.cos(2*np.pi*f1*tot_t+phi1)
            I2 = A2*np.cos(2*np.pi*f2*tot_t+phi2)

            for i in range(1,half_cycles,2):
                idx = round(i*pulse_t/dt)
                break_idxs = np.arange(idx,idx+break_t/dt+1).astype('int')
    #            print('idx',idx)
    #            print('end idx',break_idxs[-1])

                if break_idxs[-1] > tot_t.size-1:
                    break_idxs = np.arange(idx,tot_t.size-1).astype('int')


                break_ts = tot_t[break_idxs]
                if low_f:
                    I1[break_idxs] = 0
                    I2[break_idxs] = 0
                else:
                    I2[break_idxs] = A2*np.cos(2*np.pi*f1*break_ts+phi2) # I2 switch


        elif duty_cycle == 1:  # continous TI

            I1 = A1*np.cos(2*np.pi*f1*tot_t+phi1)
            I2 = A2*np.cos(2*np.pi*f2*tot_t+phi2)


        elif duty_cycle == 0:  # no signal
            f1 = 0
            f2 = 0
            I1 = A1*np.cos(2*np.pi*f1*tot_t+phi1)
            I2 = A2*np.cos(2*np.pi*f2*tot_t+phi2)


        else:
            I1 = np.zeros([1,tot_t.size])[0]
            I2 = np.zeros([1,tot_t.size])[0]

            start_idx = 0

            for i in range(0,half_cycles):
                if not i%2:
                    end_idx = start_idx + round(pulse_t/dt) + 1 - int(i!=0) + int(i+1==half_cycles)
                    pulse_idxs = np.arange(start_idx, end_idx).astype('int')
    #                print('pulse',pulse_idxs[0],pulse_idxs[-1])

                    if pulse_idxs[-1] > tot_t.size-1:
                        pulse_idxs = np.arange(start_idx,tot_t.size-1).astype('int')

                    pulse_ts = tot_t[pulse_idxs];

                    if i == 0:
                        phase1 = 0
                        phase2 = 0

                    else:
                        # offset forwards - smaller phases
                        time_shift = i/2*(pulse_t-break_t)
                        phase1 = 2*np.pi*f1*time_shift
                        phase2 = 2*np.pi*f2*time_shift


                        # scaling factor
                        c1 = np.floor(phase1/(2*np.pi));
                        c2 = np.floor(phase2/(2*np.pi));

                        # phase in the range -2pi to +2pi
                        phase1 = phase1 - c1*2*np.pi;
                        phase2 = phase2 - c2*2*np.pi;


                        # offset backwards - easier
        #                     time_offset = cycle_t*(i-1)/2
        #                     phase1 = -2*np.pi*f1*time_offset
        #                     phase2 = -2*np.pi*f2*time_offset

                    # add phase offset
                    I1[pulse_idxs] = A1*np.cos(2*np.pi*f1*pulse_ts+phi1+phase1)
                    I2[pulse_idxs] = A2*np.cos(2*np.pi*f2*pulse_ts+phi2+phase2)

                    start_idx = end_idx


                else:   # set f2 to f1 during break

                    end_idx = start_idx + round(break_t/dt) + int(i+1==half_cycles)
                    break_idxs = np.arange(start_idx, end_idx).astype('int')
    #                print('break',break_idxs[0],break_idxs[-1])

                    if break_idxs[-1] > tot_t.size-1:
                        break_idxs = np.arange(start_idx,tot_t.size-1).astype('int')

                    break_ts = tot_t[break_idxs]

                    if low_f:
                        I1[break_idxs] = 0
                        I2[break_idxs] = 0
                    else:
                        I1[break_idxs] = A1*np.cos(2*np.pi*f1*break_ts+phi1)
                        I2[break_idxs] = A2*np.cos(2*np.pi*f1*break_ts+phi2) # use f1

                    start_idx = end_idx


        ## Add ramps
        if ramp_up_t:
            ramp_up_size = round(ramp_up_t/dt)
            ramp_vec = np.arange(0,1,1/ramp_up_size)

            I1[0:ramp_vec.size] = ramp_vec*I1[0:ramp_vec.size]
            I2[0:ramp_vec.size] = ramp_vec*I2[0:ramp_vec.size]

        if ramp_down_t:
            ramp_down_size = round(ramp_down_t/dt)
            ramp_vec = np.arange(0,1,1/ramp_down_size)[::-1]

            I1[-ramp_vec.size:] = ramp_vec*I1[-ramp_vec.size:]
            I2[-ramp_vec.size:] = ramp_vec*I2[-ramp_vec.size:]


        ## Add pre and post times
        if pre_t or post_t:
            pre_zeros = np.zeros([1,int(pre_t/dt)])[0]
            post_zeros = np.zeros([1,int(post_t/dt)])[0]

            I1 = np.concatenate([pre_zeros,I1,post_zeros])
            I2 = np.concatenate([pre_zeros,I2,post_zeros])
            tot_t = np.arange(0,stim_t+post_t+pre_t+dt,dt)
            tot_t = tot_t[0:len(I1)] # sometimes one index more?


        ## Display created signal
        I = I1+I2

        if plot_on:
            plt.figure()
            plt.plot(tot_t,I)
            plt.xlim([0,tot_t[-1]])

            if h_on:
                y_h = hilbert(I)
                y_env = np.abs(y_h)
                plt.plot(tot_t, y_env, 'r')


    #PWM TI
    else:

        if carrier_f == pulse_f:
            f2 = pulse_f
        else:
            f2 = carrier_f+pulse_f

        if f1 == 0:
            A1=0
            A2 = 2*A2
        duty_cycle_2=duty_cycle
        PW_1=duty_cycle*(1/f1)
        PW_2=duty_cycle_2*(1/f2)
        sq1=A1/2*signal.square(2*np.pi*f1*tot_t,PW_1/(1/f1))
        sq2=-A1/2*signal.square(2*np.pi*f1*(tot_t-PW_1),PW_1/(1/f1))
        sq_1 = np.add(sq1,sq2)
        sq3=A2/2*signal.square(2*np.pi*f2*tot_t,PW_2/(1/f2))
        sq4=-A2/2*signal.square(2*np.pi*f2*(tot_t-PW_2),PW_2/(1/f2))
        sq_2 = np.add(sq3,sq4)
        I = sq_1+sq_2


        if ramp_up_t:
            ramp_up_size = round(ramp_up_t/dt)
            ramp_vec = np.arange(0,1,1/ramp_up_size)

            I[0:ramp_vec.size] = ramp_vec*I[0:ramp_vec.size]


        if ramp_down_t:
            ramp_down_size = round(ramp_down_t/dt)
            ramp_vec = np.arange(0,1,1/ramp_down_size)[::-1]

            I[-ramp_vec.size:] = ramp_vec*I[-ramp_vec.size:]


        if pre_t or post_t:
            pre_zeros = np.zeros([1,int(pre_t/dt)])[0]
            post_zeros = np.zeros([1,int(post_t/dt)])[0]

            I = np.concatenate([pre_zeros,I,post_zeros])
            tot_t = np.arange(0,stim_t+post_t+pre_t+dt,dt) #Redefine tot_t


        if plot_on:
            plt.figure()
            plt.plot(tot_t,I)
            plt.xlim([0,tot_t[-1]])

            if h_on:
                y_h = hilbert(I)
                y_env = np.abs(y_h)
                plt.plot(tot_t, y_env, 'r')

    return I, tot_t



if __name__ == '__main__':
    carrier_f = 1000
    pulse_f = 10
    dc = .5
    A=100/2
    I = createPulsedSignal(carrier_f, pulse_f, stim_t=4000, duty_cycle=dc, dt=0.01,
                   phi1=0, phi2=np.pi, A1=A, A2=A, pre_t=50, post_t=50,
                   ramp_up_t=500, ramp_down_t=500, plot_on=1, waveform_mode=2, h_on=0)
