from typing import List

import pytest

from cleanse_and_tokenize import run_cleanse_tokenize
from generate_raw_names import run_name_generation
from train_model import run_train_model
from utils import encode, decode, pad_token_list, pad_tokens


def test_encode_decode():
    # Assert that decoder returns same input that was passed to encoder
    sample_input = ["hello world", "this is", "", "a", "test"]
    encoded = encode(sample_input)
    decoded = decode(encoded)

    assert decoded == sample_input


def test_pad_tokens():
    test_cases = [[1], [1, 2], [1, 2, 3, 4, 5]]
    target_len = 4
    padded_cases = pad_token_list(test_cases, target_len)
    print(padded_cases)
    for case in padded_cases:
        assert len(case) == target_len


def test_pad_encode_decode():
    target_len = 10
    sample_input = ["hello"]

    encoded = encode(sample_input)
    padded: List[List[int]] = pad_token_list(encoded, target_len)
    decoded = decode(padded)

    assert decoded == ["¬¬¬¬¬" + "hello"]


def integration_validate_pipeline():
    run_name_generation()
    run_cleanse_tokenize()
    run_train_model()



