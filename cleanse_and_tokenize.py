from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from random import sample
from keras.preprocessing.text import Tokenizer
import pickle
import h5py
import os
import re
import logging

invalid_letters_pattern = r"""[^a-z0-9\s\'\-\.\&]"""
multiple_spaces_pattern = r"""\s+"""
output_dir = ""
name_col = "name"
max_features = 20000
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

stats_out = True
if stats_out:
    logger.info("Custom stats logging is enabled")

stats_out = True
business_names_path = "test/data/companies_sorted.csv" #"/kaggle/input/free-7-million-company-dataset/companies_sorted.csv"
individual_names_path = "test/data/individuals_generated.csv" # "/kaggle/input/generate-fake-names/generated_full_names.csv"


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


def output_stats(tokenizer: Tokenizer, test: pd.DataFrame, train: pd.DataFrame, business_names: pd.DataFrame, individual_names: pd.DataFrame):
    list_tokenized_business = tokenizer.texts_to_sequences(business_names[name_col].tolist())
    list_tokenized_individual = tokenizer.texts_to_sequences(individual_names[name_col].tolist())
    (business_tokens_count, individual_tokens_count, shared_tokens_count) = calculate_token_counts(list_tokenized_business, list_tokenized_individual)
    logger.info(f"training set shape: {train.shape}")
    logger.info(f"test set shape: {test.shape}")
    logger.info(f"Number of individual name samples: {individual_names.count()}")
    logger.info(f"Number of business name samples: {business_names.count()}")
    logger.info(f"Number of names shared between individual and business name sets: {len(list(set(individual_names) & set(business_names)))}")
    logger.info(f"Number of distinct business tokens: {business_tokens_count}")
    logger.info(f"Number of distinct individual tokens: {individual_tokens_count}")
    logger.info(f"Number of tokens shared between business and individual datasets: {shared_tokens_count}")


if __name__ == "__main__":

    business_names_raw: pd.DataFrame = pd.read_csv(business_names_path, dtype=str)[[name_col]]
    individual_names_raw: pd.DataFrame = pd.read_csv(individual_names_path,dtype=str)[[name_col]]
    business_names = preprocess_raw_data(business_names_raw)
    individual_names = preprocess_raw_data(individual_names_raw)

    business_names["is_business"] = 1
    individual_names["is_business"] = 0

    combined_names = business_names.append(individual_names).drop_duplicates().reindex()
    train = combined_names.copy().sample(frac=0.8,random_state=42)
    test = combined_names.copy().drop(train.index).sample(frac=1).reset_index(drop=True)

    train.to_csv(output_dir + "train.csv", index = False, header=False)
    test.to_csv(output_dir + "test.csv", index = False, header=False)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(combined_names[name_col].tolist())

    pickle_out = open(output_dir + "tokenizer.pickle", "wb")
    pickle.dump(tokenizer, pickle_out)
    pickle_out.close()

    if stats_out:
        output_stats(tokenizer, test, train, business_names, individual_names)


