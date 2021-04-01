#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import librosa.display
from matplotlib import pyplot as plt

from utilities import get_progression_beat_positions, get_bar_positions, get_quarter_beat_positions, get_eighth_beat_positions


get_ipython().run_line_magic('matplotlib', 'inline')


colors = ['r','tab:orange','k','c','g','y','m','w','b','c','0.5','0.6','0.7','0.8','0.9','1.0']

root_dir = '/mnt/d/projects/bassline_extraction'
plot_dir = os.path.join(root_dir, 'figures', 'plots')

confidence_dir = os.path.join(plot_dir, 'F0 Confidence Plots')
spectral_plots_dir = os.path.join(plot_dir, 'Spectral Plots')
time_freq_dir = os.path.join(plot_dir, 'Time-Frequency Plots')

spectrogram_dir = os.path.join(spectral_plots_dir, 'spectrograms')
note_spec_dir = os.path.join(spectral_plots_dir, 'notes')
spectral_comparison_dir = os.path.join(spectral_plots_dir, 'comparisons')

#quantization_spec_dir = os.path.join(spectral_comparison_dir, 'quantization')
confidence_filtering_spec_dir = os.path.join(spectral_comparison_dir, 'confidence_filtering')
algorithm_comparison_spec_dir = os.path.join(spectral_comparison_dir, 'algorithm_comparison')

algorithm_comparison_raw_dir = os.path.join(algorithm_comparison_spec_dir, 'raw_outputs')
algorithm_comparison_confidence_dir = os.path.join(algorithm_comparison_spec_dir, 'confidence_filtered')
algorithm_comparison_quantized_dir = os.path.join(algorithm_comparison_spec_dir, 'quantized')

wave_spec_dir = os.path.join(time_freq_dir, 'wave_spectrograms')
note_wave_spec_dir = os.path.join(wave_spec_dir, 'notes')


def beat_plotting(title):
    """
    Makes the beat-grid plottable.
    """
    
    beat_positions = get_progression_beat_positions(title)
    beat_positions -= beat_positions[0]
    bar_positions = get_bar_positions(beat_positions)
    beat_positions_plotting = [val for idx,val in enumerate(beat_positions) if idx%4]
    quarter_beat_positions = [val for idx,val in enumerate(get_quarter_beat_positions(beat_positions)) if idx%4]  
    #eighth_beat_positions = get_eighth_beat_positions(beat_positions)
        
    return bar_positions, beat_positions_plotting, quarter_beat_positions


def form_beat_grid_waveform(title, audio_array, fs, ax):
    """
    Plots the bar, beat and quarter beats on a given waveform plt.ax
    """

    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)

    librosa.display.waveplot(audio_array, sr=fs, ax=ax)
    ax.vlines(beat_positions_plotting, -0.9, 0.9, alpha=0.8, color='r',linestyle='dashed', linewidths=3)
    ax.vlines(quarter_beat_positions, -0.7, 0.7, alpha=0.8, color='k',linestyle='dashed', linewidths=3)
    ax.vlines(bar_positions, -1.1, 1.1, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax.set_xlim([-0.05, (len(audio_array)/fs)+0.05])
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)


def form_beat_grid_spectrogram(title, ax, spectrogram, fs, hop_length):
    """
    Plots the bar, beat and quarter beats on a given spectrogram plt.ax
    """

    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax.vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax.vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax.set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    display_frequencies = np.array([0,32,48,64,96,128,256,512])
    ax.yaxis.set_ticks(display_frequencies)
    ax.set_yticklabels(display_frequencies, fontsize=12)  
    ax.set_ylim([-5,512]) 
    ax.yaxis.label.set_size(14)
    ax.xaxis.label.set_size(14)

     

def form_pitch_track(F0_estimate, ax, color='b', label=''):
    """
    Plots the F0_estimate on a given plt.ax
    """

    time_axis, F0 = F0_estimate
    markerline, stemlines, baseline = ax.stem(time_axis, F0, basefmt=" ", label=label)
    markerline.set_markerfacecolor(color)
    markerline.set_markersize(8)
    stemlines.set_linewidth(0)


def save_function(save, plot_dir, title, plot_title='', default_title=''):
    """
    Saves the plot to a given directory with a given default plot title or the provided plot title.
    """
  
    if save:
        if not plot_title:
            plt.savefig(os.path.join(plot_dir,'{}-{}.png'.format(title, default_title)))
        else:
            plt.savefig(os.path.join(plot_dir,'{}-{}.png'.format(title, plot_title)))  
 

def plot_spec(title, spectrogram, fs, hop_length, F0_estimate=None, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    form_beat_grid_spectrogram(title, ax, spectrogram, fs, hop_length)
    
    if F0_estimate:   
        form_pitch_track(F0_estimate, ax)

    save_function(save, spectrogram_dir, title, plot_title=plot_title, default_title='Spectrogram')
               
    plt.show()


def plot_wave_spec(title, audio_array, spectrogram, fs, hop_length, F0_estimate=None, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    
    if F0_estimate:   
        form_pitch_track(F0_estimate, ax[0]) 

    form_beat_grid_waveform(title, audio_array, fs, ax[1])

    save_function(save, wave_spec_dir, title, plot_title=plot_title, default_title='Wave_Spec')
            
    plt.show()


def plot_wave_spec_notes(title, audio_array, spectrogram, fs, hop_length, bassline_notes, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    
    for i, note_dict in enumerate(list(bassline_notes.values())):

        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            form_pitch_track((note_dict['time'], note_dict['frequency']), ax[0], color=colors[i], label=note) 
        
    ax[0].legend(loc=1, fontsize=12) 
    
    form_beat_grid_waveform(audio_array, fs, ax[1])

    save_function(save, note_wave_spec_dir, title, plot_title=plot_title, default_title='Wave_Spec_Notes')
           
    plt.show()


def plot_spec_notes(title, spectrogram, fs, hop_length, bassline_notes, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)

    for i, note_dict in enumerate(list(bassline_notes.values())):

        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            form_pitch_track((note_dict['time'], note_dict['frequency']), ax[0], color=colors[i], label=note) 
        
    plt.legend(loc=1, fontsize=12)

    save_function(save, note_spec_dir, title, plot_title=plot_title, default_title='Spectrogram_Notes')  
                      
    plt.show()    
    
    
def plot_note_comparison(title, spectrogram, fs, hop_length, F0_estimate, bassline_notes, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    
    for i, note_dict in enumerate(list(bassline_notes.values())):

        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            form_pitch_track((note_dict['time'], note_dict['frequency']), ax[0], color=colors[i], label=note) 
        
    ax[0].legend(loc=1, fontsize=12) 

    form_beat_grid_spectrogram(title, ax[1], spectrogram, fs, hop_length)
    
    form_pitch_track(F0_estimate, ax[1])
    
    save_function(save, spectral_comparison_dir, title, plot_title=plot_title, default_title='Note_Comparison')  
                   
    plt.show()       


def plot_confidence_filtering_effect(title, spectrogram, fs, hop_length, F0_estimate, pitch_track, save=False, plot_title=''):

    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    ax[0].set_title('Initial Estimation', fontsize=16)
    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimate, ax[0])

    ax[1].set_title('Confidence Level Filtered', fontsize=16)
    form_beat_grid_spectrogram(title, ax[1], spectrogram, fs, hop_length)
    form_pitch_track(pitch_track, ax[1]) 

    save_function(save, confidence_filtering_spec_dir, title, plot_title=plot_title, default_title='Confidence_Filtering')

    plt.show()


def plot_algorithm_comparison_raw(title, spectrogram, fs, hop_length, F0_estimates, estimator_names, save=False, plot_title=''):
    """
    Plots the comparison of two F0 estimator algorithm's raw outputs.
    """
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimates[0], ax[0])

    form_beat_grid_spectrogram(title, ax[1], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimates[1], ax[1])

    save_function(save, algorithm_comparison_raw_dir, title, plot_title=plot_title, default_title='Raw_Outputs')
              
    plt.show()


def plot_confidence(title, confidence, save=False, plot_title=''):

    histogram, bin_edges = np.histogram(confidence, bins=50)

    fig, ax = plt.subplots()

    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(center, histogram, align='center', width=width, label='Histogram')
    ax.vlines(confidence.mean(), 0, histogram.max()+20, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax.vlines(confidence.mean()+confidence.std()*np.array([-1, 1])/2, 0, histogram.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax.set_title('{} - ({})'.format('pYIN',title))
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Number of Occurances')
    
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    
    ax.legend()

    save_function(save, confidence_dir, title, plot_title=plot_title, default_title='Confidence_Level')

    plt.close()
    #plt.show()


def plot_compared_confidences(title, confidence_crepe, confidence_pyin, save=False):

    histogram_crepe, bin_edges_crepe = np.histogram(confidence_crepe, bins=25)
    histogram_pyin, bin_edges_pyin = np.histogram(confidence_pyin, bins=25)

    fig, ax = plt.subplots(2, 1, figsize=(10,8), sharex=True, sharey=True,constrained_layout=True)

    fig.suptitle('{} Confidence Level Comparisons'.format(title))

    width = 0.7 * (bin_edges_crepe[1] - bin_edges_crepe[0])
    center = (bin_edges_crepe[:-1] + bin_edges_crepe[1:]) / 2
    ax[0].bar(center, histogram_crepe, align='center', width=width, label='Histogram')
    ax[0].vlines(confidence_crepe.mean(), 0, histogram_crepe.max()+50, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax[0].vlines(confidence_crepe.mean()+confidence_crepe.std()*np.array([-1, 1])/2, 0, histogram_crepe.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax[0].set_title('CREPE')
    ax[0].set_xlabel('Confidence')
    ax[0].set_ylabel('#Occurances')
    ax[0].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    ax[0].legend()

    width = 0.7 * (bin_edges_pyin[1] - bin_edges_pyin[0])
    center = (bin_edges_pyin[:-1] + bin_edges_pyin[1:]) / 2
    ax[1].bar(center, histogram_pyin, align='center', width=width, label='Histogram')
    ax[1].vlines(confidence_pyin.mean(), 0, histogram_pyin.max()+50, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax[1].vlines(confidence_pyin.mean()+confidence_pyin.std()*np.array([-1, 1])/2, 0, histogram_pyin.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax[1].set_title('pYIN')
    ax[1].set_xlabel('Confidence')
    ax[1].set_ylabel('#Occurances')
    ax[1].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    ax[1].legend()  


    if save:
        plt.savefig(os.path.join(confidence_dir, '{}_confidence_comparisons.png'.format(title)))
        plt.close()
    else:
        plt.show()