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
        threshold = mean - (std/2)
    else:
        assert threshold < 1.0 and threshold > 0, 'Threshold must be inside (0, 1)'

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
    #print('Mean of the confidence levels: {:.3f}\nStandard Deviation: {:.3f}'.format(mean, std))

    if threshold == 'none':
        threshold = 0
    elif threshold == 'mean':
        threshold = mean
    elif threshold == 'mean_reduced':
        threshold = mean - (std/2)
    else:
        assert threshold < 1.0 and threshold > 0, 'Threshold must be inside (0, 1)'

    F0_filtered = np.array(confidence_filter(F0, confidence, threshold))

    time_axis = np.arange(len(F0)) * (hop_length/fs)

    return (time_axis, F0), (time_axis, F0_filtered)


def confidence_filter(F0, confidence, threshold):
	"""
	Silences the time instants where the model confidence is below the given threshold.
	"""

	return [f if confidence[idx] >= threshold else 0.0 for idx, f in enumerate(F0)]


def yin_crepe(title, bassline, fs, frame_length, save=False):
        
    time_axis, F0, confidence_crepe, _ = crepe_predict(bassline, fs, viterbi=True)
           
    #F0_filtered = np.array(confidence_filter(F0, confidence_crepe, threshold))
    
    F0_estimate_crepe = (time_axis, F0)
    #pitch_track_crepe = (time_axis, F0_filtered)
    
    
    hop_length = int(frame_length/4) # 4 by default   
    F0, _, confidence_pyin = pyin(bassline, sr=fs, frame_length=frame_length, fmin=31.0, 
                            fmax=130.0, fill_na=0.0)

    #F0_filtered = np.array(confidence_filter(F0, confidence_pyin, threshold))
    
    time_axis = np.arange(len(F0)) * (hop_length/fs) 
    
    F0_estimate_pyin = (time_axis, F0)
    #pitch_track_pyin = (time_axis, F0_filtered) 
    
      
    plot_compared_confidences(title, confidence_crepe, confidence_pyin, save=save)
    
    return [F0_estimate_crepe, F0_estimate_pyin] #, [pitch_track_crepe, pitch_track_pyin]


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


def create_pitch_histograms(F0, boundaries, epsilon, scale_frequencies):  
    """
    For each time interval, quantizes the frequencies and creates a pitch histogram.
    
    Parameters:
    -----------
        F0_estimate (array): freq
        epsilon (int): freq_bound for frequency quantization
        boundaries: (int or np.ndarray) Number of samples each time interval has. 
                    6 samples correspond to 1/8th beat and 12 1/4th for 120<=BPM<=130.
                    you can also provide the boundary of each region separately in an ndarray
                    
    Returns:
    --------
        pitch_histograms: (list) a list of pitch histogram Counters()
    """
    
    assert (isinstance(boundaries, int) or isinstance(boundaries, np.ndarray)), (
        'provide a single interval length or a list of voiced region boundaries')
    
    if isinstance(boundaries, int): # create the boundaries with uniform time quantization
        boundaries = [[i*boundaries, (i+1)*boundaries] for i in range(int(len(F0)/boundaries))]

    pitch_histograms = []        
    for start, end in boundaries:

        interval_pitch_histogram = single_pitch_histogram(F0[start:end], epsilon, scale_frequencies)
        pitch_histograms.append(interval_pitch_histogram)
            
    return pitch_histograms


def get_majority_pitches(chunk_dicts):
    """
    Takes the majority pitch in an interval's pitch histogram.
    DEAL WITH EQUAL VOTES!!!!!!!!!!!!!!!
    """
    
    return [max(hist, key=lambda k: hist[k]) for hist in chunk_dicts]


# def group_UNK():
    """
    If there are repeating UNK frequencies, labels them. (sharp notes etc.)
    """


def sample_and_hold(bin_freqs, N_samples):
    """
    Repeats each frequency N_samples times correspondingly.
    """
    
    if isinstance(N_samples, int): # uniform sample rate
        return [f for f in bin_freqs for _ in range(N_samples)]
    
    else: # varying sample length 
        return [sample for idx, val in enumerate(N_samples) for sample in sample_and_hold([bin_freqs[idx]], val)]  


def calcRegionBounds(bool_array):
    '''
    Returns the lower and upper bounds of contiguous regions.

    Parameters
    ==========
    bool_array  1-D Binary numpy array
    '''
    assert(bool_array.dtype == 'bool' )
    idx = np.diff(np.r_[0, bool_array, 0]).nonzero()[0]
    assert(len(idx)%2 == 0)
    return np.reshape(idx, (-1,2))


def find_voiced_regions(F0):
    """
    From a given F0 array, finds the voiced regions' boundaries and returns them with corresponding lengths and indices.
    """
    
    voiced_boundaries = calcRegionBounds(F0 != 0.0)
    
    return get_region_information(voiced_boundaries)


def get_region_information(boundaries):
    """
    Packs the boundaries, lengths, and the corresponding indices in a tupple.
    """

    lengths = np.diff(boundaries, 1).flatten().tolist()

    indices = [x for start, end in boundaries for x in np.arange(start, end)]

    return (boundaries, lengths, indices)


def uniform_quantization(pitch_track, segments, scale_frequencies, epsilon=4):
    """
    Uniformly quantizes each given segment independently.
    """

    # Find the voiced regions 
    boundaries, lengths, indices = segments

    # Form the Pitch Histogram and Do Majority Voting for each eegion independently
    pitch_histograms = create_pitch_histograms(pitch_track[1], boundaries, epsilon, scale_frequencies)
    majority_pitches = get_majority_pitches(pitch_histograms)    

    #Quantize Each Region
    quantized_non_zero_frequencies = sample_and_hold(majority_pitches,  lengths)
    assert len(indices) == len(quantized_non_zero_frequencies), 'Hold lengths do not match'

    # replace regions with quantized versions
    quantized_pitches = pitch_track[1].copy()
    np.put(quantized_pitches, indices, quantized_non_zero_frequencies)

    pitch_track_quantized = (pitch_track[0], quantized_pitches) # (time, freq) 

    return  pitch_track_quantized


def uniform_voiced_region_quantization(pitch_track, scale_frequencies, epsilon=4):
    """
    Finds the voiced regions, and uniformly quantizes each region in frequency using majority voting.
    """

    voiced_regions = find_voiced_regions(pitch_track[1])   

    pitch_track_quantized = uniform_quantization(pitch_track, voiced_regions, scale_frequencies, epsilon)

    return  pitch_track_quantized


def find_closest_quarter_beat(time, quarter_beat_positions):
    
    delta = np.abs(time - quarter_beat_positions)
    delta_min = np.min(delta)
        
    return np.where(delta==delta_min)[0][0]


def find_closest_note(beat_time, time_axis):
    
    delta = np.abs(beat_time - time_axis)
    delta_min = np.min(delta)
    
    return np.where(delta==delta_min)[0][0]    


def segment_voiced_regions(time_axis, voiced_boundaries, quarter_beat_positions):
    """
    Segments voiced regions if they have proper length, otherwise categorizes them.
    """
    
    quarter_beat_positions -= quarter_beat_positions[0] # start from 0

    segment_boundaries = [] # good regions' segmentation boundaries
    okay_region_boundaries = [] # okayish regions
    bad_region_boundaries = [] # bad regions

    # for each voiced region
    for onset_idx, offset_idx in voiced_boundaries:

        if offset_idx-onset_idx >= 8: # if region length is suitable for segmentation

            # get the times 
            onset_time = time_axis[onset_idx]
            offset_time = time_axis[offset_idx-1] # end boundary not included


            # find the closest quarter beats to the onset and offset times and 
            # make sure that onset starts before the beat and the offest ends after a beat
            start_beat_idx = find_closest_quarter_beat(onset_time, quarter_beat_positions)   
            if quarter_beat_positions[start_beat_idx] < onset_time:            
                start_beat_idx += 1

            start_beat_time = quarter_beat_positions[start_beat_idx] 

            end_beat_idx = find_closest_quarter_beat(offset_time, quarter_beat_positions)
            if quarter_beat_positions[end_beat_idx] > offset_time:
                end_beat_idx -= 1

            end_beat_time = quarter_beat_positions[end_beat_idx]


            # segmentation starts with the onset idx and the closest quarter beat's corresponding idx    
            b1 = onset_idx
            b2 = find_closest_note(start_beat_time, time_axis)

            if not b1 == b2: # onset on quarter beat
                segment_boundaries.append([b1, b2])
                
            # segment between each qbeat inside the adjusted region 
            for k in np.arange(start_beat_idx, end_beat_idx+1):

                if k == end_beat_idx:
                    break

                # get the qbeat times
                start_beat_time = quarter_beat_positions[k] 
                end_beat_time = quarter_beat_positions[k+1]

                # get the corresponding indices in the pitch track
                b1 = find_closest_note(start_beat_time, time_axis)
                b2 = find_closest_note(end_beat_time, time_axis)

                segment_boundaries.append([b1, b2])
                
            # segmentation finishes with the final qbeat and the offset idx
            b1 = find_closest_note(quarter_beat_positions[k], time_axis)
            b2 = offset_idx-1

            if not b1 == b2: # onset on quarter beat
                segment_boundaries.append([b1, b2])

        elif offset_idx-onset_idx >= 4: # if region can be uniformly quantized

            okay_region_boundaries.append([onset_idx, offset_idx])

        else: # if region will be erased
               
            bad_region_boundaries.append([onset_idx, offset_idx])

    segments = get_region_information(np.array(segment_boundaries))
    okay_regions = get_region_information(np.array(okay_region_boundaries))
    bad_regions = get_region_information(np.array(bad_region_boundaries))

    return segments, okay_regions, bad_regions


def adaptive_voiced_region_quantization(pitch_track, quarter_beat_positions, scale_frequencies, epsilon=4):
    """
    
    """

    # Find the voiced regions
    voiced_boundaries, region_lengths, voiced_indices = find_voiced_regions(pitch_track[1])

    # segment the voiced regions
    segments, okay_regions, bad_regions = segment_voiced_regions(pitch_track[0], 
                                                                    voiced_boundaries, 
                                                                    quarter_beat_positions)


    # Uniformly quantize each segmented region independtly 
    pitch_track_quantized = uniform_quantization(pitch_track, segments, scale_frequencies, epsilon)

    # Uniformly quantize okay regions, without segmentation
    pitch_track_quantized = uniform_quantization(pitch_track_quantized, okay_regions, scale_frequencies, epsilon)

    # Silence bad regions
    pitch_track_quantized = region_silencer(pitch_track_quantized, bad_regions)

    return pitch_track_quantized


def onset_offset_merger(pitch_track_quantized, segments):

    


def region_silencer(pitch_track, bad_regions):
    """
    Zeroes out given regions.
    """

    # get the indices to be silenced
    _, _, indices = bad_regions

    silenced_pitch_track = np.array([0.0 if idx in indices else f for idx, f in enumerate(pitch_track[1])])

    return (pitch_track[0], silenced_pitch_track)


def extract_notes(F0_estimate, notes, scale_frequencies):
    """
    Creates a dictionary of (time,pitch) mappings from an F0_estimate tupple
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