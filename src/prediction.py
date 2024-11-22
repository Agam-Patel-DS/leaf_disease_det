import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import json
from IPython.display import FileLink
from PIL import Image
import io

class prediction_config:
  def __init__(self,params):
    self.model=params["model"]
    self.class_indices=params["class_indices"]

class predict:
  def __init__(self,prediction_config):
    self.model=prediction_config.model
    self.class_indices=prediction_config.class_indices

  def predict_image(self,image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    model = tf.keras.models.load_model(self.model)

    # Make the prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Load the class indices
    with open(self.class_indices, 'r') as f:
      class_indices = json.load(f)
    predicted_disease=""
    # Get the predicted disease
    for key, value in class_indices.items():
      if value == predicted_class:
        predicted_disease = key
        break
    print(f"The disease is {predicted_disease}")
    return predicted_disease






