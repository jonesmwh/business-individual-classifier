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
from utils import list_to_csv, init_logger, encode, pad_token_list, load_config
import random


config = load_config()

invalid_letters_pattern = r"""[^a-z0-9\s\'\-\.\&]"""
multiple_spaces_pattern = r"""\s+"""
input_dir = config["rel_paths"]["raw_data_root"].get(str)
output_dir = config["rel_paths"]["cleansed_data_root"].get(str)
interim_dir = config["rel_paths"]["interim_debug_root"].get(str)
output_path = output_dir + config["cleanse_and_tokenize"]["output_filename"].get(str)
max_features = config["cleanse_and_tokenize"]["encoding_space_size"].get(int)
min_length = config["cleanse_and_tokenize"]["min_length"].get(int)

name_col = "name"
y_col = "is_business"
tokenized_length = config["cleanse_and_tokenize"]["tokenized_length"].get(int)
logger = init_logger()

stats_out = True
write_out_interim_data = True

if stats_out:
    logger.info("Custom stats logging is enabled")

stats_out = True
business_names_path = input_dir + config["cleanse_and_tokenize"]["businesses_filename"].get(str)
individual_names_path = input_dir + config["cleanse_and_tokenize"]["individuals_filename"].get(str)


def remove_invalid_characters(input_text: str) -> str:

    remove_invalid_letters = re.sub(invalid_letters_pattern, " ", input_text)
    remove_multiple_spaces = re.sub(multiple_spaces_pattern, " ", remove_invalid_letters)
    text_valid_chars_only = remove_multiple_spaces.strip()

    return text_valid_chars_only


def introduce_misspellings(names: pd.DataFrame) -> pd.DataFrame:
    names[name_col] = names[name_col].map(lambda original_name: remove_random_characters(name=original_name, no_chars=2))
    return names


def remove_random_characters(name: str, no_chars: int) -> str:
    if no_chars == 0:
        return name
    else:
        updated_name = remove_char(name)
        return remove_random_characters(updated_name, no_chars - 1)


def remove_char(original: str) -> str:
    if len(original) > 1:
        index = random.randint(0, len(original)-1)
        less_one_char = original[:index] + original[index+1:]
        return less_one_char
    else:
        return original


def preprocess_raw_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    preprocessed = data_frame.\
        applymap(str).\
        dropna().\
        apply(lambda x: x.str.lower()).\
        applymap(lambda raw_string: remove_invalid_characters(raw_string))

    mask = (preprocessed[name_col].str.len() >= min_length) & (preprocessed[name_col] != "")
    preprocessed = preprocessed.loc[mask].drop_duplicates()

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
    logger.info(f"Number of individual name samples: {individual_names[name_col].count()}")
    logger.info(f"Number of business name samples: {business_names[name_col].count()}")
    logger.info(f"Number of names shared between individual and business name sets: {shared_names_count}")
    logger.info(f"Number of distinct business tokens: {business_tokens_count}")
    logger.info(f"Number of distinct individual tokens: {individual_tokens_count}")
    logger.info(f"Number of tokens shared between business and individual datasets: {shared_tokens_count}")


def generate_model_input(data: pd.DataFrame) -> (np.ndarray, List[str]):
    name_list = data[name_col]

    list_tokenized = encode(name_list)

    X = np.array(pad_token_list(list_tokenized, tokenized_length))
    y = data[y_col]
    return X, y


def write_interim_out_csv(X_train, X_test, X_test_mutated, y_train: List[str], y_test: List[str], train, test, test_mutated):
    train.to_csv(interim_dir + "train_untokenized.csv", index=False, header=False)
    test.to_csv(interim_dir + "test_untokenized.csv", index=False, header=False)
    test_mutated.to_csv(interim_dir + "test_mutated_untokenized.csv", index=False, header=False)
    np.savetxt(interim_dir + "X_train.csv", X_train, delimiter=",", fmt="%i")
    np.savetxt(interim_dir + "X_test.csv", X_test, delimiter=",", fmt="%i")
    np.savetxt(interim_dir + "X_test_mutated.csv", X_test_mutated, delimiter=",", fmt="%i")
    list_to_csv(y_train, interim_dir + "y_train.csv")
    list_to_csv(y_test, interim_dir + "y_test.csv")


def run_cleanse_tokenize():
    business_names_raw: pd.DataFrame = pd.read_csv(business_names_path, dtype=str)[[name_col]]
    individual_names_raw: pd.DataFrame = pd.read_csv(individual_names_path,dtype=str)[[name_col]]
    business_names = preprocess_raw_data(business_names_raw)
    individual_names = preprocess_raw_data(individual_names_raw)

    business_names[y_col] = 1
    individual_names[y_col] = 0

    combined_names = business_names.append(individual_names, ignore_index=True).drop_duplicates().dropna().reset_index(drop=True)
    train, test = train_test_split(combined_names, test_size=0.1, random_state=42)
    test_mutated = introduce_misspellings(test.copy())

    (X_train, y_train) = generate_model_input(train)
    (X_test, y_test) = generate_model_input(test)
    (X_test_mutated, y_test_mutated) = generate_model_input(test_mutated)

    if write_out_interim_data:
        write_interim_out_csv(X_train, X_test, X_test_mutated, y_train, y_test, train, test, test_mutated)

    if stats_out:
        output_stats(test, train, business_names, individual_names)

    pickle_out = open(output_path, "wb")
    pickle.dump((X_train, y_train, X_test, y_test, tokenized_length, max_features), pickle_out, protocol=4)
    pickle_out.close()


if __name__ == "__main__":
    run_cleanse_tokenize()
