import pickle
import confuse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.engine.saving import load_model
import pandas as pd

from utils import load_config


if __name__ == "__main__":
    config = load_config()

    cleansed_data_dir = config["rel_paths"]["cleansed_data_root"].get(str)
    interim_dir = config["rel_paths"]["interim_debug_root"].get(str)
    model_dir = config["rel_paths"]["models_root"].get(str)
    model_path = model_dir + config["train_model"]["best_model_filename"].get(str)
    training_data_path = cleansed_data_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)

    (X_train, y_train, X_test, y_test, tokenized_length, max_features) = pickle.load(open(training_data_path, "rb"))

    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_bool = list(np.round(y_pred).flat)

    # Print f1, precision, and recall scores
    precision = precision_score(y_test, y_pred_bool, average="macro")
    recall = recall_score(y_test, y_pred_bool, average="macro")
    f1 = f1_score(y_test, y_pred_bool, average="macro")
    print("--- test data ---")
    print(f"precision score: {precision}")
    print(f"recall score: {recall}")
    print(f"F1 score: {f1}")

    X_test_mutated = np.loadtxt(interim_dir + "X_test_mutated.csv", int, delimiter=",")
    y_pred = model.predict(X_test_mutated)
    y_pred_bool = list(np.round(y_pred).flat)
    precision = precision_score(y_test, y_pred_bool, average="macro")
    recall = recall_score(y_test, y_pred_bool, average="macro")
    f1 = f1_score(y_test, y_pred_bool, average="macro")
    print("--- mutated test data (with spelling mistakes) ---")
    print(f"precision score: {precision}")
    print(f"recall score: {recall}")
    print(f"F1 score: {f1}")

