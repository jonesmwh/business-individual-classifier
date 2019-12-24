import csv
import logging
import confuse
from typing import List

default_config_path = "config/config_default.yaml"


def load_config(path: str = default_config_path):
    config = confuse.Configuration("business-individual-classifier", __name__)
    config.set_file(path)
    return config


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = init_logger()


def list_to_csv(list: List[str], output_path: str, header: str = ""):
    with open(output_path, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header != "": writer.writerow([header])
        for element in list:
            writer.writerow([element])
    writeFile.close()


def pad_token_list(list_tokenized: List[List[int]], total_length: int) -> List[List[int]]:
    #Apply pad_tokens to list of token lists
    padded = []
    for item in list_tokenized:
        padded.append(pad_tokens(item, total_length))
    return padded


def pad_tokens(tokenized: List[int], total_length: int) -> List[int]:
    # Pad/concatenate list of tokens, to equal total_length
    token_count = len(tokenized)
    diff = token_count - total_length
    if diff == 0:
        padded = tokenized
    if diff < 0:
        padded = [ord("¬") for blanks in range(-diff)]+tokenized
    if diff > 0:
        logger.warning(
            f"pad_sequences function detected token sequence of {token_count} items. "
            f"This is greater than max length of {total_length}.")
        padded = tokenized[-total_length:]
    return padded


def decode(encoded: List[List[int]]) -> List[str]:
    # decode utf8 to character
    return ["".join([chr(char) for char in name]) for name in encoded]


def encode(name_list: List[str]) -> List[List[int]]:
    # encode character to utf8
    return [[ord(char) for char in name] for name in name_list]

