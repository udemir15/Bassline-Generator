import os
import torch

def load_state_dict(model_name):

    model_path = os.path.join('model_checkpoints', model_name+'.pt')

    checkpoint = torch.load(model_path)

    return checkpoint['model_state_dict']
