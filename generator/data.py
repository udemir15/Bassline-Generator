import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import replace_with_dict, one_hot_encode


def get_data(M=8, test_size=0.2, gam='minor'):
    minor_df = pd.read_csv(
        f"/kuacc/users/udemir15/ELEC491/bassline_transcription/data/datasets/[28, 51]/TechHouse_bassline_representations_min_M{M}.csv", header=None)
    major_df = pd.read_csv(
        f"/kuacc/users/udemir15/ELEC491/bassline_transcription/data/datasets/[28, 51]/TechHouse_bassline_representations_maj_M{M}.csv", header=None)

    all_data = pd.concat((minor_df.iloc[:, 1:].astype(
        int), major_df.iloc[:, 1:].astype(int)))

    minor_data = minor_df.values[:, 1:].astype(int)
    major_data = major_df.values[:, 1:].astype(int)

    notes = np.unique(all_data)
    vocab = np.arange(len(notes))

    n2v_mapping = dict(zip(notes, vocab))
    v2n_mapping = dict(zip(vocab, notes))

    vocab_size = len(n2v_mapping)

    minor_data = replace_with_dict(minor_data, n2v_mapping).astype(int)

    major_data = replace_with_dict(major_data, n2v_mapping).astype(int)

    if gam == 'minor':
        source_data = minor_data
    elif gam == 'major':
        source_data = major_data
    else:
        source_data = pd.concat((minor_data, major_data))

    X_train, X_test, y_train, y_test = train_test_split(
        source_data, source_data, test_size=test_size, random_state=42)

    return X_train, X_test, one_hot_encode(y_train, vocab_size), one_hot_encode(y_test, vocab_size), vocab_size
