from bs4 import BeautifulSoup

from brushes.nlp.encoders import (
    ContractionsRemover,
    DataFrameSelector,
    StopwordsRemover,
    TextLowerCaser,
    TextPunctuationRemover,
)

from sklearn.pipeline import Pipeline

from sumukha.config import PUNCTUATION, preprocess_path

import re


def encode_labels(label):
    if label == 'positive':
        return 1
    return 0


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def clean_data(data):

    data.dropna(inplace=True)

    data['review'] = data['review'].apply(strip_html)
    data['review'] = data['review'].apply(remove_special_characters)
    data['sentiment'] = data['sentiment'].apply(encode_labels)

    reviews, *labels = DataFrameSelector(['review', 'sentiment']).transform(data)

    features_encoder = Pipeline([
        ('remove_punctuation', TextPunctuationRemover(punctuation=PUNCTUATION, replace_with_space=True)),
        ('remove_stopwords', StopwordsRemover()),
        ('lowercaser', TextLowerCaser()),
        ('lemmatizer', ContractionsRemover()),
    ])

    data['review'] = features_encoder.transform(reviews.copy())

    data['review'].to_csv(preprocess_path + 'data.txt', index=False, header=False)

    return data

