# Ship-Detection-in-Optical-Satellite-Imagery

This project proposes a processing pipeline for
automated detection of ships in optical satellite imagery. Dataset
from Kaggle’s Airbus Ship Detection Challenge is used for
the project. This dataset has 190,000, 768 x 768 pixel images
with complex backgrounds of clouds, shore lines, waves and
ship-wakes. The large field of view images in the dataset
makes saliency detection a necessary first step. The saliency
map provides several candidates that can be tested with a
classifier. Several feature descriptors are explored, compared
and combined to determine the best method for ship detection.
Principle Component Analysis (PCA) is employed to reduce
the feature size and its results are also investigated. This
project achieves a 82% classification score with classical pattern
recognition methods.

Contributors: Gaurav Shalin, Saad Rasheed Abbasi and Souvik Roy
