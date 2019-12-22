from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from random import sample
import pickle
import h5py
import os
import re
from utils import list_to_csv, init_logger, encode, pad_seq

invalid_letters_pattern = r"""[^a-z0-9\s\'\-\.\&]"""
multiple_spaces_pattern = r"""\s+"""
output_dir = "test/data/"
name_col = "name"
y_col = "is_business"
max_features = 20000
maxlen = 30
logger = init_logger()

stats_out = True
write_out_interim_data = True

if stats_out:
    logger.info("Custom stats logging is enabled")

stats_out = True
business_names_path = output_dir + "companies_sorted.csv" #"/kaggle/input/free-7-million-company-dataset/companies_sorted.csv"
individual_names_path = output_dir + "individuals_generated.csv" # "/kaggle/input/generate-fake-names/generated_full_names.csv"


def remove_invalid_characters(input_text: str) -> str:

    remove_invalid_letters = re.sub(invalid_letters_pattern, " ", input_text)
    remove_multiple_spaces = re.sub(multiple_spaces_pattern, " ", remove_invalid_letters)
    text_valid_chars_only = remove_multiple_spaces.strip()

    return text_valid_chars_only


def preprocess_raw_data(data_frame: pd.DataFrame) -> pd.DataFrame:

    preprocessed =  data_frame .\
        applymap(str).\
        dropna().\
        apply(lambda x: x.str.lower()).\
        applymap(lambda raw_string: remove_invalid_characters(raw_string))

    preprocessed = preprocessed[preprocessed[name_col] != ""].drop_duplicates()
    return preprocessed


def calculate_token_counts(list_tokenized_business: List[List[int]], list_tokenized_individual: List[List[int]]):

    business_tokens = set([item for sublist in list_tokenized_business for item in sublist])
    individual_tokens = set([item for sublist in list_tokenized_individual for item in sublist])
    shared_tokens_count = len(list(set(business_tokens) & set(individual_tokens)))
    return (len(business_tokens), len(individual_tokens), shared_tokens_count)


def output_stats(test: pd.DataFrame, train: pd.DataFrame, business_names: pd.DataFrame, individual_names: pd.DataFrame):

    list_tokenized_business = encode(business_names[name_col])
    list_tokenized_individual = encode(individual_names[name_col])
    (business_tokens_count, individual_tokens_count, shared_tokens_count) = calculate_token_counts(list_tokenized_business, list_tokenized_individual)
    shared_names = list(set(individual_names) & set(business_names))
    shared_names_count = len(shared_names)
    logger.info(f"training set shape: {train.shape}")
    logger.info(f"test set shape: {test.shape}")
    logger.info(f"Number of individual name samples: {individual_names.count()}")
    logger.info(f"Number of business name samples: {business_names.count()}")
    logger.info(f"Number of names shared between individual and business name sets: {shared_names_count}")
    logger.info(f"Number of distinct business tokens: {business_tokens_count}")
    logger.info(f"Number of distinct individual tokens: {individual_tokens_count}")
    logger.info(f"Number of tokens shared between business and individual datasets: {shared_tokens_count}")


def generate_model_input(data: pd.DataFrame) -> (np.ndarray, List[str]):
    name_list = data[name_col]

    list_tokenized = encode(name_list)

    X = pad_seq(list_tokenized, maxlen)
    y = data[y_col]
    return X, y


def write_interim_out_csv(X_train, X_test, y_train: List[str], y_test: List[str]):

    np.savetxt(output_dir + "X_train.csv", X_train, delimiter=",", fmt="%i")
    np.savetxt(output_dir + "X_test.csv", X_test, delimiter=",", fmt="%i")
    list_to_csv(y_train, output_dir + "y_train.csv")
    list_to_csv(y_test, output_dir + "y_test.csv")


if __name__ == "__main__":

    business_names_raw: pd.DataFrame = pd.read_csv(business_names_path, dtype=str)[[name_col]]
    individual_names_raw: pd.DataFrame = pd.read_csv(individual_names_path,dtype=str)[[name_col]]
    business_names = preprocess_raw_data(business_names_raw)
    individual_names = preprocess_raw_data(individual_names_raw)

    business_names[y_col] = 1
    individual_names[y_col] = 0

    combined_names = business_names.append(individual_names).drop_duplicates().dropna().reindex()
    train = combined_names.copy().sample(frac=0.8, random_state=42)
    test = combined_names.copy().drop(train.index).sample(frac=1).reset_index(drop=True)

    train.to_csv(output_dir + "train_untokenized.csv", index=False, header=False)
    test.to_csv(output_dir + "test_untokenized.csv", index=False, header=False)

    (X_train, y_train) = generate_model_input(train)
    (X_test, y_test) = generate_model_input(test)

    if write_out_interim_data: write_interim_out_csv(X_train, X_test, y_train, y_test)


    pickle_out = open(output_dir + "train_test.pickle", "wb")
    pickle.dump((X_train, y_train, X_test, y_test), pickle_out)
    pickle_out.close()

    if stats_out:
        output_stats(test, train, business_names, individual_names)
