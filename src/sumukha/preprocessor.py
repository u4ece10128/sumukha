# -*- coding: utf-8 -*-
import os

from brushes.io.filesystem import save_obj

from sumukha import config
from sumukha.preprocessing.input_preprocessor import clean_data

import pandas as pd


def preprocessor(root_path='./', input_path=config.input_file_path,
                 preprocess_path=config.input_preprocess_path):
    # Create preprocess dir if it doesn't exists
    if not os.path.exists(preprocess_path):
        os.makedirs(preprocess_path)

    # Dataset
    input_path = root_path + input_path

    data = pd.read_csv(input_path + os.listdir(input_path)[0], keep_default_na=False)

    preprocessed_dataset = {
        'features': clean_data(data),
    }
    preprocess_path = root_path + preprocess_path

    # Save preprocessed dataset and encoders to data/preprocess
    save_obj(preprocessed_dataset, preprocess_path, 'dataset')
    print(f'Preprocessed files from "{input_path}" moved in "{preprocess_path}"')
    return preprocessed_dataset


def run_preprocess(root_path, input_path, preprocess_path):
    return preprocessor(root_path=root_path, input_path=input_path, preprocess_path=preprocess_path)


if __name__ == '__main__':
    preprocessor()
