import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import replace_with_dict, one_hot_encode


def get_data(M=8, test_size=0.2, gam='min', notes=list(range(26))):
    assert gam in ['min', 'maj', 'all'], "gam must be min or maj!"

    if gam != 'all':
        df = pd.read_csv(
            f"/kuacc/users/udemir15/ELEC491/bassline_transcription/data/datasets/[28, 51]/TechHouse_bassline_representations_{gam}_M{M}.csv", header=None)
    else:
        minor_df = pd.read_csv(
                f"/kuacc/users/udemir15/ELEC491/bassline_transcription/data/datasets/[28, 51]/TechHouse_bassline_representations_min_M{M}.csv", header=None)
        major_df = pd.read_csv(
                f"/kuacc/users/udemir15/ELEC491/bassline_transcription/data/datasets/[28, 51]/TechHouse_bassline_representations_maj_M{M}.csv", header=None)
        df = pd.concat((minor_df, major_df))
    
    #df.iloc[:,1:] = df.iloc[:,1:].astype(int)

    vocab = np.arange(len(notes))

    n2v_mapping = dict(zip(notes, vocab))
    #v2n_mapping = dict(zip(vocab, notes))

    vocab_size = len(n2v_mapping)

    #minor_data = replace_with_dict(minor_data, n2v_mapping).astype(int)

    #major_data = replace_with_dict(major_data, n2v_mapping).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df.values, df.values, test_size=test_size, random_state=42)

    return X_train[:,0].tolist(), X_train[:,1:].astype(int), X_test[:,0].tolist(), X_test[:,1:].astype(int), one_hot_encode(y_train[:,1:], vocab_size), one_hot_encode(y_test[:,1:], vocab_size), vocab_size
