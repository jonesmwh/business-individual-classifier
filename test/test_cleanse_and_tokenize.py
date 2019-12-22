import pytest

from utils import encode, decode, pad_seq


def test_encode_decode():
    # Assert that decoder returns same input that was passed to encoder
    sample_input = ["hello world", "this is", "", "a", "test"]
    encoded = encode(sample_input)
    decoded = decode(encoded)

    assert decoded == sample_input

def test_pad_tokens():
    test_cases = [[1],[1,2],[1,2,3,4,5]]
    target_len = 4
    padded_cases = pad_seq(test_cases, target_len)
    print(padded_cases)
    for case in padded_cases:
        assert len(case) == target_len
