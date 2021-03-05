#!/usr/bin/env python
# coding: utf-8

import os

def init_folders():

    if not os.path.exists('data/bassline_extraction'):
        os.mkdir('data/bassline_extraction')

    if not os.path.exists('data/bassline_extraction/beat_grid'):
        os.mkdir('data/bassline_extraction/beat_grid')

    if not os.path.exists('data/bassline_extraction/beat_grid/beat_positions'):
        os.mkdir('data/bassline_extraction/beat_grid/beat_positions')

    if not os.path.exists('data/bassline_extraction/beat_grid/aligned_beat_positions'):
        os.mkdir('data/bassline_extraction/beat_grid/aligned_beat_positions')

    if not os.path.exists('data/bassline_extraction/beat_grid/bad_examples'):
        os.mkdir('data/bassline_extraction/beat_grid/bad_examples')


    if not os.path.exists('data/bassline_extraction/choruses/initial_chorus_estimates'):
        os.mkdir('data/bassline_extraction/choruses/initial_chorus_estimates')

    if not os.path.exists('data/bassline_extraction/choruses/aligned_choruses'):
        os.mkdir('data/bassline_extraction/choruses/aligned_choruses')


    if not os.path.exists('data/bassline_extraction/basslines'):
        os.mkdir('data/bassline_extraction/basslines')

    if not os.path.exists('data/bassline_extraction/basslines/processed'):
        os.mkdir('data/bassline_extraction/basslines/processed')

    if not os.path.exists('data/bassline_extraction/basslines/unprocessed'):
        os.mkdir('data/bassline_extraction/basslines/unprocessed')
