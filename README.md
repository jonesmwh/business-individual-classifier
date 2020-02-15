# business-individual-classifier

## Why?

Accurately classify the names of businesses/organisations and people/individuals, even when input data has misspellings and typos that were not present in the training data.

When parsing real-world poor quality data, sometimes necessary to classify whether the entity in a "name" field corresponds to an organisation or a person. Common approaches to classify names in this context are:
  - Rules based systems e.g. searching for terms such as "Mrs." or "Ltd."
  - ML classifier models which learn the probability of word tokens taken from a labelled set of business/individual names

Neither of these approaches are resiliant when dealing with poor data quality e.g. misspellings and typos, as this effectively introduces new tokens which were not present in the training data - e.g. the model has no way to infer that the business "Tom's Co**n**puters" is similar to the phrase "Jack's Co**m**puters" that was included in the training set.

This model uses a character/byte level deep learning model, meaning it can infer that unseen words such as "Co**n**puters" are similar to the word "Computers" that it has trained on. This also has the advantage that the model could be trained on words with non-latin characters with little modification, resulting in a global name classification model. 

## Technical Details

Classifies strings corresponding to names into businesses or individuals.

Uses a character/byte-level deep learning model (LSTM trained in Tensorflow). 
The model was trained on approximately 7 million unique business names from LinkedIn, and 7 million distinct individuals names from IMDB.

It achieved a 95% accuracy (F1 score) based upon a held-out validation set from the same distribution. 

When typos/spelling mistakes are introduced into the test set, simulated by deleting two characters at random from each test item (without changing the training data/retraining with misspelled data), the model achieves with 86% accuracy (F1 score).

A character/byte-level model (tokens correspond to utf8 code for each character in the input string) was selected because:

- The number of tokens is much smaller than a more typical word-level model (approximately 200 in our case, Vs hundreds of thousands in a word level model). This resulted in a 30x reduction in trained model size, and significant simplification in preprocessing, expected to improve in-use performance.
- There is no 'out of lexicon' problem for words not in the training set
- Similar word roots, and misspellings not included in the training set can still be classified due to the existing context of other characters
- The model could be extended quite simply to use non-latin alphabet characters

## Usage

- cleanse_and_tokenize.py	: Cleanse and tokenize pre-existing raw data from IMDB and Linkeding busienss names (see Data section below)
- train_model.py : Train LSTM deep lerning model
- validate_model.py	: Calculate accuracy metrics (F1 score, precision, recall) for trained model based upon held out validation dataset
- classify_names.py	: Example script for using trained model to classify a list of names
- generate_raw_names.py	: Generate dummy raw data sets for testing purposes

## References

### Data

#### Faker artificial name generator (for testing):
- https://pypi.org/project/Faker/

#### Linkedin Business names :
- https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset
  
#### IMDB actor names:
- https://www.kaggle.com/ashirwadsangwan/imdb-dataset

### Methodology

#### Byte-level encodings: 
- https://medium.com/analytics-vidhya/https-medium-com-tomkenter-why-care-about-byte-level-seq2seq-models-in-nlp-26bcf05dd7d3

- http://www.tomkenter.nl/pdf/kenter_byte-level_2018.pdf
