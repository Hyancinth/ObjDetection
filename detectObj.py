from yolo3 import*
import csv
import tensorflow as tf
import config

def main():
    try:
        modelPath = config.modelPath
        tf.keras.models.load_models(modelPath)
        
    except FileNotFoundError:
        print("\n[In Progress] Making Model")
        Model = make_yolov3_model()
        print("\n[Success] Model made")
