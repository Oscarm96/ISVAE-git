import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import dct



def normal(x, mean, sd, log = False):
    if log:
        return -0.5*((x-mean)**2/sd)
    else:
        return torch.exp(-0.5*((x-mean)**2/sd))
    

def synthetic_2():
    n_sequences = 1000
    list_labels = []
    X = np.ones((n_sequences, 600))

    for j in range(n_sequences):
        n_slots = 3
        n, p = 1, .5  # number of trials, probability of each trial
        s = np.random.binomial(n, p, n_slots)

        mu_phase, sigma_phase = 0, 0
        mu_amp, sigma_amp = 0.05, 0.25
        mu_fq, sigma_fq = 0, 0

        fs = 600

        if s[0] == 0 and s[1] == 0 and s[2] == 0:
            label = 1
            f_1 = 80
            f_2 = 130
            f_3 = 495
        if s[0] == 0 and s[1] == 0 and s[2] == 1:
            label = 2
            f_1 = 180
            f_2 = 390
            f_3 = 596
        if s[0] == 0 and s[1] == 1 and s[2] == 0:
            label = 3
            f_1 = 80
            f_2 = 130
            f_3 = 230
            f_4 = 390
        if s[0] == 0 and s[1] == 1 and s[2] == 1:
            label = 4
            f_1 = 180
            f_2 = 230
            f_3 = 430
            f_4 = 530
        if s[0] == 1 and s[1] == 0 and s[2] == 0:
            label = 5
            f_1 = 80
            f_2 = 180
            f_3 = 315
            f_4 = 495
        if s[0] == 1 and s[1] == 0 and s[2] == 1:
            label = 6
            f_1 = 230
            f_2 = 390
            f_3 = 495
            f_4 = 596
        if s[0] == 1 and s[1] == 1 and s[2] == 0:
            label = 7
            f_1 = 130
            f_2 = 230
            f_3 = 430
            f_4 = 530
        if s[0] == 1 and s[1] == 1 and s[2] == 1:
            label = 8
            f_1 = 80
            f_2 = 315
            f_3 = 495
            f_4 = 596

        list_labels = list_labels + [label]

        x = np.arange(fs)

        noise_phase = np.random.normal(mu_phase, sigma_phase, 1)
        noise_amplitude = np.random.normal(mu_amp, sigma_amp, x.shape)
        noise_fq = np.random.normal(mu_fq, sigma_fq, 1)

        if label == 1 or label == 2:
            y_1 = noise_amplitude + 1 * np.cos(2 * np.pi * (f_1 + noise_fq) * (x / fs) + noise_phase)
            noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
            y_2 = (noise_amplitude + np.cos(2 * np.pi * (f_2 + noise_fq) * (x / fs) + noise_phase))
            noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
            y_3 = (noise_amplitude + np.cos(2 * np.pi * (f_3 + noise_fq) * (x / fs) + noise_phase))

            signal = y_1 + y_2 + y_3
        else:
            y_1 = noise_amplitude + 1 * np.cos(2 * np.pi * (f_1 + noise_fq) * (x / fs) + noise_phase)
            noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
            y_2 = (noise_amplitude + np.cos(2 * np.pi * (f_2 + noise_fq) * (x / fs) + noise_phase))
            noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
            y_3 = (noise_amplitude + np.cos(2 * np.pi * (f_3 + noise_fq) * (x / fs) + noise_phase))
            noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
            y_4 = (noise_amplitude + np.cos(2 * np.pi * (f_4 + noise_fq) * (x / fs) + noise_phase))

            signal = y_1 + y_2 + y_3 + y_4

        signal = np.expand_dims(signal, axis=0)
        X[j, :] = signal

    return X, list_labels


def label_sample_unique(labels, X):

    uni = np.unique(labels)
    labels_np = np.asarray(labels)

    signals = np.ones((uni.shape[0], X.shape[1]))
    labels_sig = []

    for i in range(uni.shape[0]):
        mask = labels_np == uni[i]

        signals[i, :] = X[mask][0, :]
        labels_sig = labels_sig + [labels_np[mask][0]]

    return signals, labels_sig


def plots_time(labels, X):
    
    signals, label_sig = label_sample_unique(labels, X)
    n_classes = signals.shape[0]

    n_row = 2
    n_col = int(n_classes/2)
    
    plt.figure(figsize=(20, 15))
    for i in range(signals.shape[0]):
        plt.subplot(n_row, n_col, i + 1)
        plt.title(label_sig[i])
        plt.plot(signals[i, :])
    plt.show()

    return n_classes

def plots_freq(labels, X_dct):
    signals, label_sig = label_sample_unique(labels, X_dct)

    n_classes = signals.shape[0]

    n_row = 2
    n_col = int(n_classes / 2)
  
    plt.figure(figsize=(20, 15))
    for i in range(signals.shape[0]):
        plt.subplot(n_row, n_col, i + 1)
        plt.title(label_sig[i])
        plt.plot(signals[i, 1:])

    
    psd_dct = np.mean(X_dct ** 2, axis=0)

    X_psd_class = np.zeros((n_classes, X_dct.shape[1] - 1))
    plt.figure(figsize=(20, 15))
    for pos, i in enumerate(np.unique(labels)):
        plt.subplot(n_row, n_col, pos + 1)
        plt.title(i)
        mask_psd = labels == i
        X_class = X_dct[mask_psd, 1:]
        X_class_psd = np.mean(X_class ** 2, axis=0)
        X_psd_class[pos, :] = X_class_psd
        plt.plot(X_psd_class[pos, 1:])

    return n_classes, psd_dct, X_psd_class




def load_data(): 
    
    #opciones dataset = [HAR, HB_sound, syn, syn2, insect, HAR_ext, electronic_dev, kitchen, HAR_act, SODA, steps]
   
    X, labels = synthetic_2()

    X_dct = np.ones_like(X)
    for i in range(X.shape[0]):
       X_dct[i, :] = dct(X[i, :], 1)
   
    n_classes = plots_time(labels, X)
    n_classes, psd_dct, X_psd_class = plots_freq(labels, X_dct)


    return X, X_dct, labels, n_classes, psd_dct, X_psd_class
