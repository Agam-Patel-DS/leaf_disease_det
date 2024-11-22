import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import streamlit as st

from src.prediction import prediction_config, predict
pred_params=yaml.safe_load(open("params.yaml"))["prediction"]

config=prediction_config(pred_params)

pred=predict(config)
pred.predict_image("/content/drive/MyDrive/leaf_disease_detection/sample_images/potato.jpeg")