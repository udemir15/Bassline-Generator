#!/usr/bin/env python
# coding: utf-8

import os
from matplotlib import pyplot as plt
import IPython.display as ipd
import librosa
import numpy as np

# directories
root_dir = '/mnt/d/projects/bassline_extraction'

data_dir = os.path.join(root_dir,'data')

bassline_extraction_dir = os.path.join(data_dir,'bassline_extraction')

beat_grid_dir = os.path.join(bassline_extraction_dir,'beat_grid')
beat_positions_dir = os.path.join(beat_grid_dir, 'beat_positions')
aligned_beat_positions_dir = os.path.join(beat_grid_dir, 'aligned_beat_positions')
bad_examples_dir = os.path.join(beat_grid_dir,'bad_examples')

chorus_dir = os.path.join(bassline_extraction_dir, 'choruses')
initial_chorus_estimates_dir = os.path.join(chorus_dir, 'initial_chorus_estimates')
aligned_choruses = os.path.join(chorus_dir, 'aligned_choruses')

bassline_dir = os.path.join(bassline_extraction_dir, 'basslines')
processed_bassline_dir = os.path.join(bassline_dir, 'processed')
unprocessed_bassline_dir = os.path.join(bassline_dir, 'unprocessed')

clip_dir = os.path.join(data_dir,'audio_clips')

directories = [bassline_extraction_dir, beat_grid_dir, beat_positions_dir, beat_positions_dir,
                aligned_beat_positions_dir, bad_examples_dir, chorus_dir, initial_chorus_estimates_dir,
               aligned_choruses, bassline_dir, processed_bassline_dir, unprocessed_bassline_dir]


def init_folders():
    
    for directory in directories:
        if not os.path.exists(directory): 
            os.mkdir(directory)
        

def print_plot_play(x, Fs, text=''):
    
    print('%s\n' % (text))
    print('Fs = %d, x.shape = %s, x.dtype = %s' % (Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs)) #,normalize=False
    
    
def load_audio(track_title):
    """
    Loads experiment outputs from numpy arrays.
    """
    
    chorus = np.load(aligned_choruses+'/'+track_title+'.npy')
    processed_bassline = np.load(processed_bassline_dir+'/'+track_title+'.npy')
    unprocessed_bassline = np.load(unprocessed_bassline_dir+'/'+track_title+'.npy')    
    
    return chorus, processed_bassline, unprocessed_bassline


def load_track(track_title, fs):
    """
    Loads a track given title.
    """
    return librosa.load(os.path.join(clip_dir,track_title+'.mp3'),sr=fs)
    
def inspect_audio_outputs(track_title):
    chorus, processed_bassline, unprocessed_bassline = load_audio(track_title)
    
    print('\t\t{}\n'.format(track_title))
    print_plot_play(chorus, 44100, 'Aligned Chorus')
    print_plot_play(processed_bassline, 44100, 'Processed Bassline')
    print_plot_play(unprocessed_bassline, 44100, 'Unprocessed Bassline')
    

def get_track_scale(title, track_dicts, scales):
    
    key, scale_type = track_dicts[title]['Key'].split(' ')
    scale_frequencies = scales[key][scale_type]['frequencies']
    notes = [note+'0' for note in scales[key][scale_type]['notes']]
    notes += [note+'1' for note in scales[key][scale_type]['notes']]     
        
    return notes, scale_frequencies['0'] + scale_frequencies['1']    


def search_idx(query, track_titles):
    found = False
    for idx, title in enumerate(track_titles):
    
        if title == query:
            print(idx)
            found = True

    if not found:
        print("Track couldn't found!")


def create_frequency_bins(fs, n_fft): 
    bin_width = (fs/2) / n_fft
    frequency_bins = np.arange(0, int((n_fft/2)+1))*bin_width
    return frequency_bins, bin_width

        
def get_beat_positions(title):
    return np.load(beat_positions_dir+'/'+title+'.npy')
    
    
def get_progression_beat_positions(title):
    return np.load(aligned_beat_positions_dir+'/'+title+'.npy')


def get_bar_positions(beat_positions):    
    return [val for idx,val in enumerate(beat_positions) if not idx%4]


def get_quarter_beat_positions(beat_positions):
    quarter_beats = []
    for i in range(len(beat_positions)-1):
        for qb in np.linspace(beat_positions[i],beat_positions[i+1], 4, endpoint=False):
            quarter_beats.append(qb)
            
    return quarter_beats


def get_eighth_beat_positions(beat_positions):
    eighth_beats = []
    for i in range(len(beat_positions)-1):
        for qb in np.linspace(beat_positions[i],beat_positions[i+1], 8, endpoint=False):
            eighth_beats.append(qb)
            
    return eighth_beats
