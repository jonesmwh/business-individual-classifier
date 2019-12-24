import numpy as np
import pickle
from typing import List

import confuse
from keras.engine.saving import load_model

from utils import load_config, encode, pad_tokens, pad_token_list

config = load_config()

cleansed_data_dir = config["rel_paths"]["cleansed_data_root"].get(str)
model_dir = config["rel_paths"]["models_root"].get(str)
model_path = model_dir + config["train_model"]["best_model_filename"].get(str)
training_data_path = cleansed_data_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)

model = load_model(model_path)


def classify(names: List[str], model):
    cleansed_name = np.array(pad_token_list(encode(names), 120))
    predictions = model.predict(cleansed_name)
    for p in predictions:
        print(p)
        print("business" if int(p.round(0)) == 1  else "individual")


if __name__ == "__main__":
    name = ["matt jones", "sam hicks","mott junes", "smith compating","rob aitken","harry potter","laura ashley ltd"]
    classify(name, model)
