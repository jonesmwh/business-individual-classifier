import pickle
import confuse
from keras.engine.saving import load_model

from utils import load_config


if __name__ == "__main__":
    config = load_config()

    cleansed_data_dir = config["rel_paths"]["cleansed_data_root"].get(str)
    model_dir = config["rel_paths"]["models_root"].get(str)
    model_path = model_dir + config["train_model"]["best_model_filename"].get(str)
    training_data_path = cleansed_data_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)

    (X_train, y_train, X_test, y_test, tokenized_length, max_features) = pickle.load(open(training_data_path, "rb"))

    model = load_model(model_path)

    validation =model.evaluate(X_test,y_test)

    print(validation)