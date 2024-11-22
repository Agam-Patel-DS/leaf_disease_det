import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

import os
from src.preprocess import preprocess_config, preprocess
import yaml
params=yaml.safe_load(open("params.yaml"))["preprocess"]

config=preprocess_config(params)
prep=preprocess(config)
prep.preprocess_data()
