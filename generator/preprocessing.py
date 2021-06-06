import numpy as np


def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k, ar, sorter=sidx)]]


def one_hot_encode(arr, encode_length):
    return np.eye(encode_length)[arr.astype(np.uint)]


def replace_continue(arr, cont_token):
    for r in arr:
        for idx, el in enumerate(r[1:]):
            if el == cont_token:
                r[idx + 1] = r[idx]
