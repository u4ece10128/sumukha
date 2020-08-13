import os

input_file_path = os.environ.get('INPUT_FILE_PATH', 'data/raw/')

preprocess_path = os.environ.get('PREPROCESS_PATH', 'data/preprocess/')

PUNCTUATION = r"""!"#%&'()*+,-./:;<=>?@[\]^_`{|}~‘’"""
