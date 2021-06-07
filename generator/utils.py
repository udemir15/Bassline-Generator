import os
from datetime import datetime
import sys
sys.path.insert(0, '/scratch/users/udemir15/ELEC491/bassline_transcription')
from bassline_transcriber.transcription import NN_output_to_MIDI_file

def now():
    return '_'.join(str(datetime.now()).split('.')[0].split(' '))

def save_nn_output_to_midi(samples, name, **kwargs):
    now_str = now()
    dir_path = f'midis/{name}_{now_str}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    for idx, sample in enumerate(samples):
        NN_output_to_MIDI_file(sample, title=str(idx), output_dir=dir_path, **kwargs)