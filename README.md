# business-individual-classifier

## About

Classifies strings corresponding to names into businesses or individuals.

Uses a character/byte-level deep learning model (LSTM trained in Tensoeflow). 
The model was trained on approximately 7 million unique business names from LinkedIn, and 7 million distinct individuals names from IMDB.
It achieved a 95% accuracy based upon a held-out validation set from the same distribution.

A character/byte-level model (tokens correspond to utf8 code for each character in the input string) was selected because:

- The number of tokens is much smaller than a more typical word-level model (approximately 200 in our case, Vs hundreds of thousands in a word level model). This resulted in a 30x reduction in trained model size, and significant simplification in preprocessing, expected to improve in-use performance.
- There is no 'out of lexicon' problem for words not in the training set
- Similar word roots, and misspellings not included in the training set can still be classified due to the existing context of other characters
- The model could be extended quite simply to use non-latin alphabet characters


## References

### Data

#### Faker artificial name generator:
- https://pypi.org/project/Faker/

#### Linkedin Business names :
- https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset
  
#### IMDB actor names:
- https://www.kaggle.com/ashirwadsangwan/imdb-dataset

### Methodology

#### Tensorflow and Keras for text classification:
- https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras

#### Byte-level encodings: 
- https://medium.com/analytics-vidhya/https-medium-com-tomkenter-why-care-about-byte-level-seq2seq-models-in-nlp-26bcf05dd7d3

- http://www.tomkenter.nl/pdf/kenter_byte-level_2018.pdf
