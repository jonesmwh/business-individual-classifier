import csv
import logging
from typing import List

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
        writer = csv.writer(writeFile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header != "": writer.writerow([header])
        for element in list:
            writer.writerow([element])
    writeFile.close()


def pad_seq(list_tokenized: List[List[int]], maxlen: int) -> List[List[int]]:
    padded = []
    for item in list_tokenized:
        token_count = len(item)
        diff = token_count - maxlen
        if diff == 0:
            padded.append(item)
        if diff < 0:
            padded.append([0 for blanks in range(-diff)] + item)
        if diff > 0:
            logger.warning(f"pad_sequences function detected token sequence of {token_count} items. This is greater than max length of {maxlen}.")
            padded.append(item[-maxlen:])
    return padded


def decode(encoded: List[List[int]]) -> List[str]:
    return ["".join([chr(char) for char in name]) for name in encoded]


def encode(name_list: List[str]) -> List[List[int]]:
    return [[ord(char) for char in name] for name in name_list]

