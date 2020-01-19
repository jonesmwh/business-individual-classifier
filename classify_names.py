import numpy as np
import pickle
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import confuse
from keras.engine.saving import load_model

from utils import load_config, encode, pad_tokens, pad_token_list

config = load_config()

cleansed_data_dir = config["rel_paths"]["cleansed_data_root"].get(str)
model_dir = config["rel_paths"]["models_root"].get(str)
model_path = model_dir + config["train_model"]["best_model_filename"].get(str)
training_data_path = cleansed_data_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)

model = load_model(model_path)


def classify(names: List[str], model, verbose: bool):
    cleansed_name = np.array(pad_token_list(encode(names), 120))
    predictions = model.predict(cleansed_name)
    if verbose:
        counter = 0
        for p in predictions:
            name = names[counter]
            classification = "business" if int(p.round(0)) == 1 else "individual"
            print(f"{name} : {classification} ({round(float(p[0]),3)})")
            counter += 1
    return predictions


if __name__ == "__main__":
    name = ["matt jones", "fred smith","mott junes", "smith computing", "smith compating","rob johnson","laura smith ltd"]
    y_test = [0,0,0,1,1,0,1]
    y_pred = classify(name, model, True)
    y_pred_bool = list(np.round(y_pred).flat)

    # Print f1, precision, and recall scores
    precision = precision_score(y_test, y_pred_bool, average="macro")
    recall = recall_score(y_test, y_pred_bool, average="macro")
    f1 = f1_score(y_test, y_pred_bool, average="macro")
    print()
    print(f"precision score: {precision}")
    print(f"recall score: {recall}")
    print(f"F1 score: {f1}")

