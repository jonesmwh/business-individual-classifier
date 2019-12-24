from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py
import pickle
from utils import load_config

config = load_config()
input_dir = config["rel_paths"]["cleansed_data_root"].get(str)
output_dir = config["rel_paths"]["models_root"].get(str)

training_data_path = input_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)
final_output_path = output_dir + config["train_model"]["secondary_model_filename"].get(str)
output_path = output_dir + config["train_model"]["best_model_filename"].get(str)
history_path = output_dir + config["train_model"]["history_filename"].get(str)
embedding_size = config["train_model"]["embedding_size"].get(int)

def run_train_model():
    (X_train, y_train, X_test, y_test, tokenized_length, max_features) = pickle.load(open(training_data_path, "rb"))

    inp = Input(shape=(tokenized_length, ))
    embed_size = embedding_size
    x = Embedding(max_features, embed_size)(inp)
    x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    mc = ModelCheckpoint(output_path, monitor='val_loss', mode='min', verbose=0, save_best_only=True)

    batch_size = config["train_model"]["batch_size"].get(int)
    epochs = config["train_model"]["epochs"].get(int)
    validation_split = config["train_model"]["validation_split"].get()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[mc])

    model.summary()
    model.save(final_output_path)

    pickle_out = open(history_path, "wb")
    pickle.dump(history, pickle_out, protocol=4)
    pickle_out.close()


if __name__ == "__main__":
    run_train_model()
