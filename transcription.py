#!/usr/bin/env python
# coding: utf-8

import os
from collections import Counter

import numpy as np

from crepe import predict as crepe_predict
from librosa import pyin

from utilities import create_frequency_bins


def argmax_F0(spectrogram, fs, hop_length):
    
    time_axis = np.arange(spectrogram.shape[1]) * (hop_length/fs)
    
    frequency_bins, _ =  utils.create_frequency_bins(fs, spectrogram.shape[0])
    max_freq_bins = np.argmax(spectrogram, axis=0) # get the frequency bins with highest energy
        
    F0 = np.array([frequency_bins[idx] for idx in max_freq_bins])

    return (time_axis, F0)


def crepe_F0(audio, fs, viterbi=True, threshold='none'):
      
    time_axis, F0, confidence, _ = crepe_predict(audio, fs, viterbi=True)
    
    mean, std = np.mean(confidence), np.std(confidence)
    print('Mean of the confidence levels: {:.3f}\nStandard Deviation: {:.3f}'.format(mean, std))
    
    if threshold == 'none':
        threshold = 0
    elif threshold == 'mean':
        threshold = mean
    elif threshold == 'mean_reduced':
        threshold = mean - (std/4)
    else:
        assert threshold > 1.0 and threshold < 0, 'Threshold must be inside (0, 1)'

    F0_filtered = np.array(confidence_filter(F0, confidence, threshold))

    return (time_axis, F0), (time_axis, F0_filtered)


def pYIN_F0(audio, fs, frame_length, threshold='none'):

    #win_length =
    hop_length = int(frame_length/4) # 4 by default

    F0, _, confidence = pyin(audio,
                              sr=fs,
                              frame_length=frame_length,
                              hop_length=hop_length,
                              fmin=31.0,
                              fmax=130.0,
                              fill_na=0.0)

    mean, std = np.mean(confidence), np.std(confidence)
    print('Mean of the confidence levels: {:.3f}\nStandard Deviation: {:.3f}'.format(mean, std))

    if threshold == 'none':
        threshold = 0
    elif threshold == 'mean':
        threshold = mean
    elif threshold == 'mean_reduced':
        threshold = mean - (std/4)
    else:
        assert threshold >= 1.0 and threshold < 0, 'Threshold must be inside (0, 1)'

    F0_filtered = np.array(confidence_filter(F0, confidence, threshold))

    time_axis = np.arange(len(F0)) * (hop_length/fs)

    return (time_axis, F0), (time_axis, F0_filtered)


def confidence_filter(F0, confidence, threshold):
	"""
	Silences the time instants where the model confidence is below the given threshold.
	"""

	return [f if confidence[idx] >= threshold else 0.0 for idx, f in enumerate(F0)]


def quantize_frequency(f, epsilon, scale_frequencies):
    """
    Quantizes a frequency to the known scale frequencies using algorithm1.(Ouz)
    delta_bound ADAPTIVE OR CONSTANT ??????????
    
    Parameters:
    -----------
        f (float): frequency in Hz.
        epsilon (int): freq_bound = delta_scale/epsilon determines if quantization will happen
        scale_frequencies (list): list of the frequencies in the scale
        
    """
    
    if f: # non zero frequencies
                
        delta_array = np.abs(f - np.array(scale_frequencies)) # distances to the notes of the scale
        delta_min = np.min(delta_array) # smallest such distance

        delta_bound = np.min(np.diff(scale_frequencies))/epsilon 

        if delta_min <= delta_bound: # if there is a note closeby
            note_idx = np.where(delta_array==delta_min)[0][0] # index of the corresponding note in the scale
            f = scale_frequencies[note_idx] # quantize pitch
            
    return f 


def single_pitch_histogram(F0, epsilon, scale_frequencies):
    """
    Creates a single pitch histogram for a given interval by quantizing each frequency.
    """
   
    return Counter([quantize_frequency(f, epsilon, scale_frequencies) for f in F0])


def create_pitch_histograms(F0_estimate, boundaries, epsilon, scale_frequencies):  
    """
    For each time interval, quantizes the frequencies and creates a pitch histogram.
    
    Parameters:
    -----------
        F0_estimate (tuple of  2 arrays): (time, freq) 
        epsilon (int): freq_bound for frequency quantization
        boundaries: (int or np.ndarray) Number of samples each time interval has. 
                    6 samples correspond to 1/8th beat and 12 1/4th for 120<=BPM<=130.
                    you can also provide the boundary of each region separately in an ndarray
                    
    Returns:
    --------
        pitch_histograms: (list) a list of pitch histogram Counters()
    """
    
    assert (isinstance(boundaries, int) or isinstance(boundaries, np.ndarray)), 'provide a single interval length or a list of voiced region boundaries'
    
    if isinstance(boundaries, int): # create the boundaries for uniform time quantization
        boundaries = [[i*boundaries, (i+1)*boundaries] for i in range(int(len(F0_estimate[0])/boundaries))]

    pitch_histograms = []        
    for start, end in boundaries:

        interval_pitch_histogram = single_pitch_histogram(F0_estimate[1][start:end], epsilon, scale_frequencies)
        pitch_histograms.append(interval_pitch_histogram)
            
    return pitch_histograms


def get_majority_pitches(chunk_dicts):
    """
    Takes the majority pitch in an interval's pitch histogram.
    DEAL WITH EQUAL VOTES!!!!!!!!!!!!!!!
    """
    
    return [max(hist, key=lambda k: hist[k]) for hist in chunk_dicts]


def extract_notes(F0_estimate, notes, scale_frequencies):
    """
    Creates a dictionary of (time,pitch) mappings from a F0
    CHECK UNK FREQUENCIES TO DETECT SCALE BREAKING
    """
           
    bassline_notes = {n: {'time': [], 'frequency': []} for n in ['UNK', '-'] + notes} # note dict

    for idx, f in enumerate(F0_estimate[1]):
        
        if f: # if non-zero
            if f not in scale_frequencies:
                bassline_notes['UNK']['time'].append(F0_estimate[0][idx])
                bassline_notes['UNK']['frequency'].append(F0_estimate[1][idx])                               
            else:  
                note_idx = scale_frequencies.index(f) # index of the corresponding note in the scale
                bassline_notes[notes[note_idx]]['time'].append(F0_estimate[0][idx])
                bassline_notes[notes[note_idx]]['frequency'].append(F0_estimate[1][idx])                                
        else:      
            bassline_notes['-']['time'].append(F0_estimate[0][idx])
            bassline_notes['-']['frequency'].append(F0_estimate[1][idx])            
                                  
    return bassline_notes

def sample_and_hold(bin_freqs, N_samples):
    """
    Repeats each frequency N_samples times correspondingly.
    """
    
    if isinstance(N_samples, int): # uniform sample rate
        return [f for f in bin_freqs for _ in range(N_samples)]
    
    else: # varying sample length 
        return [sample for idx, val in enumerate(N_samples) for sample in sample_and_hold([bin_freqs[idx]], val)]  