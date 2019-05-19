#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import math
import random

def label(rle):
    if type(rle) == str:
        return 1
    
    if math.isnan(rle):
        return 0

random.seed(1000)

df = pd.read_csv('train_ship_segmentations_v2.csv')
df['path'] = df['ImageId'].map(lambda x: os.path.join('/Users/saad/train_v2', x))
df['label'] = df['EncodedPixels'].map(lambda x: label(x))

no_ships = df[df['label'] == 0]
ships = df[df['label'] == 1]

new_df = pd.concat([no_ships.sample(9000, random_state = 1000),ships.sample(1000, random_state = 1000)])
new_df = new_df.sample(frac = 1, random_state = 1000).reset_index(drop=True)

file_name = 'ships_unbalanced_S90_NS10.plk'
new_df.to_pickle(file_name)