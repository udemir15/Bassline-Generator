#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import datetime as dt
from tqdm import tqdm

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import IPython.display as ipd

# Low level Audio Processing
import librosa
import librosa.display
import soundfile as sf

# High Level Audio Processing
#from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor # Beat Tracking
from pychorus import find_and_output_chorus # Chorus Finder
#from spleeter.separator import Separator # Source Separaion

import traceback
import warnings
warnings.filterwarnings('ignore') # ignore librosa .mp3 warnings

from utilities import init_folders


class Track:
    
    def __init__(self, title, track_dicts, scales, beat_proc, tracking_proc, separator, fs=44100, N_bars=4):
        
        self.info = self.Info(title, track_dicts, scales, fs, N_bars) # Track information class
        
        self.audio = self.Audio(self.info) # The track itself, may be unnecessary
        
        self.chorus = self.Chorus(self.info) # Chorus information
        
        self.beatgrid = self.BeatGrid(self.info, self.chorus, beat_proc, tracking_proc) # Beatgrid is formed
        
        self.separator = self.Separator(self.info, self.audio, self.chorus, separator) # Bassline extractor is configured
    
    
    class Info:
        
        def __init__(self, title, track_dicts, scales, fs, N_bars):
            
            self.title = title
            self.track_dict = track_dicts[title]
            self.path = os.path.join('data','audio_clips',title+'.mp3')

            self.key = self.track_dict['Key']
            key, key_type = self.key.split(' ')
            self.scale = scales[key][key_type]

            self.BPM = int(self.track_dict['BPM'])
            self.beat_length = 60 / self.BPM # length of one beat in sec
            self.N_bars = N_bars # number of bars to consider a chorus
            self.chorus_length = N_bars * (4 * self.beat_length)
            
            self.fs = fs
            
            length_str = self.track_dict['Length'].split(':') # Tracklength from beatport
            self.length_beatport = int(length_str[0])*60+int(length_str[1])
                       
            
    class Audio:
              
        def __init__(self, info):
            
            track, sr = librosa.load(info.path, sr=info.fs)
            self.track = track
            self.info = info
            self.fs = sr # for security??  
            self.length_read = len(track)/sr # sec
            
        #def import_audio # LOAD NUMPY ARRAY OR LOAD TRACK
            
        def export_audio(self):
            np.save('data/audio_arrays/{}.npy'.format(self.info.title), self.track)
            
        def export_aligned_chorus(self):
            np.save('data/bassline_extraction/choruses/aligned_choruses/{}.npy'.format(self.info.title), self.aligned_chorus)
            
        def export_bassline(self):
            np.save('data/bassline_extraction/basslines/unprocessed/{}.npy'.format(self.info.title), self.bassline)
            
        def export_processed_bassline(self):
            np.save('data/bassline_extraction/basslines/processed/{}.npy'.format(self.info.title), self.processed_bassline)
            
        
    class Chorus:
        
        def __init__(self, info):
        
            self.info = info
        
        def find_and_output_chorus(self):
            
            chorus_start_sec = find_and_output_chorus(self.info.path, None, self.info.chorus_length)

            # If the given chorus length is too long
            while not chorus_start_sec:
                chorus_length /= 2
                chorus_start_sec = find_and_output_chorus(self.info.path, None, self.info.chorus_length)

            self.chorus_start_sec = chorus_start_sec # HOW TO PASS THIS TO ABOVE??

        def export_chorus_start(self):
            """
            Exports the initial estimation of the start time of chorus in seconds on a .npy file
            """
            np.save('data/bassline_extraction/choruses/initial_chorus_estimates/{}.npy'.format(self.info.title), np.array(self.chorus_start_sec))
        
    
    class BeatGrid:
        
        def __init__(self, info, chorus, beat_proc, tracking_proc):
            
            self.info = info
            self.chorus = chorus # CHANGE LATER???           
            self.beat_proc = beat_proc
            self.tracking_proc = tracking_proc
            self.beat_positions = self.find_beat_positions()
            self.analyze_beats()
            self.aligned_chorus_beat_positions = np.array([]) # init as empty array
              
        def find_beat_positions(self):
            
            #beat_proc = RNNBeatProcessor()
            #tracking_proc = BeatTrackingProcessor(fps=100)

            # Find the beat positions
            activations = self.beat_proc(self.info.path) # reads the input every time
            beat_positions = self.tracking_proc(activations) # SHOULD I PASS ABOVE

            return beat_positions

        def export_beat_positions(self):
            np.save('data/bassline_extraction/beat_grid/beat_positions/{}.npy'.format(self.info.title), self.beat_positions)

        def analyze_beats(self):

            beat_positions = self.beat_positions
            first_beat_position = beat_positions[0]
            number_of_beats = len(beat_positions)

            # check the returned beat lengths
            beat_lengths_extracted = np.diff(beat_positions-beat_positions[0]) # beat length of each beat
            beat_length_deviations = np.abs(beat_lengths_extracted-self.info.beat_length) # deviation from truth

            total_deviation = np.sum(beat_length_deviations) # sec
            deviation_per_beat = total_deviation / number_of_beats # sec
            deviation_pqb = deviation_per_beat/(self.info.beat_length/4) # per quarter beat

            if 100*deviation_pqb > 5:
                with open('data/bassline_extraction/beat_grid/bad_examples/{}.txt'.format(self.info.title), 'w') as outfile:
                    outfile.write(str(100*deviation_pqb))
         
        def progression_align(self):

            bar_idx = self.align_grid_to_bar()
            bar_idx = self.align_grid_to_progression(bar_idx)
            aligned_chorus_beat_positions = self.get_aligned_chorus_beats(bar_idx)
            
            self.aligned_chorus_beat_positions = aligned_chorus_beat_positions # SHOULD I PASS ABOVE???
            
            
        def align_grid_to_bar(self):
            """
            Aligns the chorus to the nearest bar assuming the track contains only uniform 4 beat long sections.

                Returns:
                --------

                   idx (int): index of the nearest bar in the track
            """

            # Find the closest beats
            indices = np.where(np.abs(self.chorus.chorus_start_sec-self.beat_positions)<self.info.beat_length)[0]

            if len(indices) > 2:
                print("Too many beats returned!")

            # Choose the even beat
            if indices[0]   % 2:
                idx = indices[1]
            elif indices[1]  % 2:
                idx = indices[0]

            # if the even idx is not a multiple of 4, moving to the next bar beginning    
            if idx % 4:
                idx += 2

            return idx
 
        def align_grid_to_progression(self, idx):
            """
            Assuming the most common 8bar progression, aligns to the nearest start of an 8bar progression.
            """

            if idx%8 :
                idx +=4

            return idx
        
        def get_aligned_chorus_beats(self, bar_idx):
            """
            Gets N_bars worth of beats from the aligned chorus.

                chorus_beat_positions (array): array of beat positions in time (seconds)

            """
            return self.beat_positions[bar_idx:bar_idx+self.info.N_bars*4+1]
            
        def export_aligned_beat_positions(self):
            np.save('data/bassline_extraction/beat_grid/aligned_beat_positions/{}.npy'.format(self.info.title), self.aligned_chorus_beat_positions)  
    

    class Separator:
        
        def __init__(self, info, audio, chorus, separator):
            
            self.separator = separator
            self.info = info
            self.audio = audio
            self.chorus = chorus

        def extract_bassline(self):
        
            if self.audio.aligned_chorus.size > 0:
                
                chorus = np.expand_dims(self.audio.aligned_chorus,1) # required for Spleeter
                prediction = self.separator.separate(chorus, audio_descriptor='') # WHAT IS AUDIO DESCRIPTOR??????

                bassline = prediction['bass'] # get the prediction
                bassline_mono = np.mean(bassline,axis=1) # convert to mono
                bassline_mono_normalized = librosa.util.normalize(bassline_mono) # normalize bassline               
                
                self.audio.bassline = bassline_mono_normalized # IS THIS CORRECT ????
                
            else:
                print("The Chorus hasn't aligned yet!")
                
        def process_bassline(self):
            
            fc = max(self.info.scale['frequencies']['1']) # cutoff frequency in Hz
            wc = fc / (self.info.fs/2) # cutoff radians
            lp = signal.firwin(5000, wc) 
            
            bassline_cut = signal.convolve(self.audio.bassline, lp) # LP filter
            bassline_cut_normalized = librosa.util.normalize(bassline_cut) # normalize
            
            self.audio.processed_bassline = bassline_cut_normalized
            
            
    def set_aligned_chorus(self): # PUT IN DATA OR WEHRE???
        
        if self.beatgrid.aligned_chorus_beat_positions.size > 0:
        
            beat_positions = self.beatgrid.aligned_chorus_beat_positions
            start_idx, end_idx = int(self.info.fs*beat_positions[0]), int(self.info.fs*beat_positions[-1])
            aligned_chorus = self.audio.track[start_idx:end_idx]

            self.audio.aligned_chorus = aligned_chorus # TO CHORUS OR TO AUDIO ????????????????
            
        else:
            print("The Beat Grid hasn't been aligned yet!.")


if __name__ == '__main__':

    init_folders()
            
    with open('data/metadata/scales_frequencies.json','r') as infile:
        scales = json.load(infile)
        
    with open('data/metadata/TechHouse_track_dicts.json','r') as infile:
        track_dicts = json.load(infile)       
    #track_titles = list(track_dicts.keys())

    with open('data/ouz_tracks.txt', 'r') as infile:
        track_titles = infile.read().split('\n')

    date = str(dt.date.today()) # for tracking experiments

    # init processors here for preventing leakage
    beat_proc = RNNBeatProcessor()
    tracking_proc = BeatTrackingProcessor(fps=100)
    separator = Separator('spleeter:4stems')

    for title in tqdm(track_titles):
        
        try:

            print('\n'+title)
        
            track = Track(title, track_dicts, scales, beat_proc, tracking_proc, separator)
            
            # export audio for later use
            #track.audio.export_audio()

            # Estimate the chorus
            track.chorus.find_and_output_chorus()

            # Export
            track.chorus.export_chorus_start()

            # Estimate the Beat Positions
            track.beatgrid.find_beat_positions()

            # Analyze and export 
            track.beatgrid.analyze_beats()
            track.beatgrid.export_beat_positions()

            # Align the Chorus using the estimated Beat Positions
            track.beatgrid.progression_align()
            track.set_aligned_chorus()

            # Export the aligned beat positions and the aligned chorus
            track.beatgrid.export_aligned_beat_positions()
            track.audio.export_aligned_chorus()

            # Extract the bassline from the aligned chorus
            track.separator.extract_bassline()
            track.separator.process_bassline()

            # Export the basslines
            track.audio.export_bassline()
            track.audio.export_processed_bassline()
            
            with open('data/bassline_extraction/completed_tracks_{}.txt'.format(date), 'a') as outfile:
                outfile.write(' \n'.join(title))            
            del track
            
        except KeyboardInterrupt:
            import sys
            sys.exit()
            pass    
        except KeyError:
            print("Key doesn't exist in track_dicts: {}\n".format(title))
            with open('data/bassline_extraction/experiment_logs/key_errors_{}.txt'.format(date), 'a') as outfile:
                outfile.write(title+'\n')            
        except Exception as ex:     
            print("There was an error on: {}".format(title))
            exception_str = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            print(exception_str+'\n')       
            with open('data/bassline_extraction/experiment_logs/exceptions_{}.txt'.format(date), 'a') as outfile:
                outfile.write(exception_str+'\n')         
            with open('data/bassline_extraction/experiment_logs/error_log_{}.txt'.format(date), 'a') as outfile:
                outfile.write(title+'\n') 