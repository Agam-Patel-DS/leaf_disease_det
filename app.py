import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import streamlit as st


st.title("Leaf Disease Detection")
from src.prediction import prediction_config, predict
pred_params=yaml.safe_load(open("params.yaml"))["prediction"]

config=prediction_config(pred_params)

pred=predict(config)

image=st.file_uploader("Upload your image!")
if st.button("Predict"):
  disease=pred.predict_image(image)
  st.write(f"The disease is {disease}")