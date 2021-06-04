import numpy as np

from sklearn.metrics import accuracy_score
import editdistance as ed

def calculate_accuracy(target_batch, input_batch):
    """shapes: (B, T) """
       
    accuracy = accuracy_score(target_batch.flatten(), input_batch.flatten(), normalize=True)
    #accuracies = [accuracy_score(t, i, normalize=True) for t, i in zip(target_batch, input_batch)]   
    #return np.sum(accuracies) / target_batch.shape[0]
    return accuracy

def lehvenstein_distance(reference_batch, hypotheses_batch):

    dist=0
    for ref, hyp in zip(reference_batch, hypotheses_batch):
        dist += ed.eval(ref, hyp)
        
    return dist/reference_batch.shape[0]