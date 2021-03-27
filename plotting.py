#!/usr/bin/env python
# coding: utf-8

import os

import librosa.display
from matplotlib import pyplot as plt

from utilities import get_progression_beat_positions, get_bar_positions, get_quarter_beat_positions, get_eighth_beat_positions


get_ipython().run_line_magic('matplotlib', 'inline')


colors = ['0.5','tab:orange','k','c','g','r','y','m','w','b','c','0.6','0.7','0.8','0.9','1.0']

root_dir = '/mnt/d/projects/bassline_extraction'
plot_dir = os.path.join(root_dir, 'figures', 'plots')

spectral_plots_dir = os.path.join(plot_dir, 'Spectral Plots')
spectrogram_dir = os.path.join(spectral_plots_dir, 'spectrograms')
note_spec_dir = os.path.join(spectral_plots_dir, 'notes')

spectral_comparison_dir = os.path.join(spectral_plots_dir, 'comparisons')

quantization_spec_dir = os.path.join(spectral_comparison_dir, 'quantization')
algorithm_comparison_spec_dir = os.path.join(spectral_comparison_dir, 'algorithm_comparison')

algorithm_comparison_raw_dir = os.path.join(algorithm_comparison_spec_dir, 'raw_outputs')
algorithm_comparison_confidence_dir = os.path.join(algorithm_comparison_spec_dir, 'confidence_filtered')
algorithm_comparison_quantized_dir = os.path.join(algorithm_comparison_spec_dir, 'quantized')

time_freq_dir = os.path.join(plot_dir, 'Time-Frequency Plots')
wave_spec_dir = os.path.join(time_freq_dir, 'wave_spectrograms')
note_wave_spec_dir = os.path.join(wave_spec_dir, 'notes')


def beat_plotting(title):
    
    beat_positions = get_progression_beat_positions(title)
    beat_positions -= beat_positions[0]
    bar_positions = get_bar_positions(beat_positions)
    beat_positions_plotting = [val for idx,val in enumerate(beat_positions) if idx%4]
    quarter_beat_positions = [val for idx,val in enumerate(get_quarter_beat_positions(beat_positions)) if idx%4]  
    #eighth_beat_positions = get_eighth_beat_positions(beat_positions)
        
    return bar_positions, beat_positions_plotting, quarter_beat_positions 


def plot_spec(title, spectrogram, fs, hop_length, F0=None, save=False, plot_title=''):
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title, fontsize=20)
        
    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax.vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax.vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax.set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.set_ylim([-5,512])
    
    if F0:   
        time_axis, F0_estimate = F0
        markerline, stemlines, baseline = ax.stem(time_axis, F0_estimate, basefmt=" ")
        markerline.set_markerfacecolor('b')
        markerline.set_markersize(8)
        stemlines.set_linewidth(0)
        
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(spectrogram_dir,'{}-Spectrogram.png'.format(title)))
        else:
            plt.savefig(os.path.join(spectrogram_dir,'{}-{}.png'.format(title,plot_title)))
        
    plt.show()


def plot_wave_spec(title, audio_array, spectrogram, fs, hop_length, F0=None, save=False, plot_title=''):
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[0].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[0].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[0].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[0].xaxis.label.set_size(15)
    ax[0].yaxis.label.set_size(15)
    ax[0].set_ylim([-5,512])
    
    if F0:   
        time_axis, F0_estimate = F0
        markerline, stemlines, baseline = ax[0].stem(time_axis, F0_estimate, basefmt=" ")
        markerline.set_markerfacecolor('b')
        markerline.set_markersize(8)
        stemlines.set_linewidth(0)
    
    librosa.display.waveplot(audio_array, sr=fs, ax=ax[1])
    ax[1].vlines(beat_positions_plotting, -0.9, 0.9, alpha=0.8, color='r',linestyle='dashed', linewidths=3)
    ax[1].vlines(quarter_beat_positions, -0.7, 0.7, alpha=0.8, color='k',linestyle='dashed', linewidths=3)
    ax[1].vlines(bar_positions, -1.1, 1.1, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[1].set_xlim([-0.05, (len(audio_array)/fs)+0.05])
    ax[1].xaxis.label.set_size(15)
    ax[1].yaxis.label.set_size(15)
    
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(wave_spec_dir,'{}-wave_spec.png'.format(title)))
        else:
            plt.savefig(os.path.join(wave_spec_dir,'{}-{}.png'.format(title,plot_title)))
            
    plt.show()


def plot_wave_spec_notes(title, audio_array, spectrogram, fs, hop_length, bassline_notes, save=False, plot_title=''):
       
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[0].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[0].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[0].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[0].xaxis.label.set_size(15)
    ax[0].yaxis.label.set_size(15)
    ax[0].set_ylim([-5,512])
    
    for i, note_dict in enumerate(list(bassline_notes.values())):
        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            markerline, stemlines, baseline = ax[0].stem(note_dict['time'], note_dict['frequency'], basefmt=" ", label=note)                           
            markerline.set_markerfacecolor(colors[i]) 
            markerline.set_markersize(12)
            stemlines.set_linewidth(0)
        
    ax[0].legend(loc=1, fontsize=12) 
    
    
    librosa.display.waveplot(audio_array, sr=fs, ax=ax[1])
    ax[1].vlines(beat_positions_plotting, -0.9, 0.9, alpha=0.8, color='r',linestyle='dashed', linewidths=3)
    ax[1].vlines(quarter_beat_positions, -0.7, 0.7, alpha=0.8, color='k',linestyle='dashed', linewidths=3)
    ax[1].vlines(bar_positions, -1.1, 1.1, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[1].set_xlim([-0.05, (len(audio_array)/fs)+0.05])
    ax[1].xaxis.label.set_size(15)
    ax[1].yaxis.label.set_size(15)
    
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(note_wave_spec_dir,'{}-wave_spec.png'.format(title)))
        else:
            plt.savefig(os.path.join(note_wave_spec_dir,'{}-{}.png'.format(title,plot_title)))
         
    plt.show()


def plot_spec_notes(title, spectrogram, fs, hop_length, bassline_notes, save=False, plot_title=''):
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    
    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax.vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax.vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax.set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.set_ylim([-5,512])
        
    for i, note_dict in enumerate(list(bassline_notes.values())):
        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            markerline, stemlines, baseline = ax.stem(note_dict['time'], note_dict['frequency'], basefmt=" ", label=note)                           
            markerline.set_markerfacecolor(colors[i]) 
            markerline.set_markersize(12)
            stemlines.set_linewidth(0)
        
    plt.legend(loc=1, fontsize=12)  
    
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(note_spec_dir,'{}-Spectrogram.png'.format(title)))
        else:
            plt.savefig(os.path.join(note_spec_dir,'{}-{}.png'.format(title,plot_title)))   
                    
    plt.show()    
    
    
def plot_note_comparison(title, spectrogram, fs, hop_length, F0_estimate, bassline_notes, save=False, plot_title=''):
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[0].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[0].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[0].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[0].xaxis.label.set_size(15)
    ax[0].yaxis.label.set_size(15)
    ax[0].set_ylim([-5,512])
    
    for i, note_dict in enumerate(list(bassline_notes.values())):
        note = list(bassline_notes.keys())[i]
        if note_dict['time']:
            markerline, stemlines, baseline = ax[0].stem(note_dict['time'], note_dict['frequency'], label=note) #                     
            markerline.set_markerfacecolor(colors[i]) 
            markerline.set_markersize(10)
            stemlines.set_linewidth(0)
        
    ax[0].legend(loc=1, fontsize=9) 
     
    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[1].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[1].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[1].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[1].xaxis.label.set_size(15)
    ax[1].yaxis.label.set_size(15)    
    ax[1].set_ylim([-5,512])
  
    time_axis, F0_estimate = F0_estimate
    markerline, stemlines, baseline = ax[1].stem(time_axis, F0_estimate, basefmt=" ")
    markerline.set_markerfacecolor('b')
    markerline.set_markersize(10)
    stemlines.set_linewidth(0) 
    
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(spectral_comparison_dir,'{}-note_comparison.png'.format(title)))
        else:
            plt.savefig(os.path.join(spectral_comparison_dir,'{}-{}.png'.format(title,plot_title)))   
                    
    plt.show()       


def plot_algorithm_comparison_raw(title, spectrogram, fs, hop_length, F0_estimates, estimator_names, save=False, plot_title=''):
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[0].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[0].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[0].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[0].xaxis.label.set_size(15)
    ax[0].yaxis.label.set_size(15)
    ax[0].set_ylim([-5,512])
    ax[0].set_title(estimator_names[0], fontsize=16)
     
    time_axis, F0_estimate = F0_estimates[0]
    markerline, stemlines, baseline = ax[0].stem(time_axis, F0_estimate, basefmt=" ")
    markerline.set_markerfacecolor('b')
    markerline.set_markersize(8)
    stemlines.set_linewidth(0)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax[1].vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='w',linestyle='dashed', linewidths=3)
    ax[1].vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3)
    ax[1].set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax[1].xaxis.label.set_size(15)
    ax[1].yaxis.label.set_size(15)
    ax[1].set_ylim([-5,512])  
    ax[1].set_title(estimator_names[1], fontsize=16) 

    time_axis, F0_estimate = F0_estimates[1]
    markerline, stemlines, baseline = ax[1].stem(time_axis, F0_estimate, basefmt=" ")
    markerline.set_markerfacecolor('b')
    markerline.set_markersize(8)
    stemlines.set_linewidth(0) 
    
    if save:
        if plot_title == '':
            plt.savefig(os.path.join(algorithm_comparison_raw_dir,'{}-wave_spec.png'.format(title)))
        else:
            plt.savefig(os.path.join(algorithm_comparison_raw_dir,'{}-{}.png'.format(title,plot_title)))
            
    plt.show()