import os

input_file_path = os.environ.get('INPUT_FILE_PATH', 'data/raw/')

input_preprocess_path = os.environ.get('PREPROCESS_PATH', 'data/preprocess/')

embeddings_path_domain = os.environ.get('DOMAIN_EM_PATH', 'embeddings/domain/')

embeddings_path_general = os.environ.get('GEN_EM_PATH', 'embeddings/general/')

embeddings_path_domain_adapted = os.environ.get('DOMAIN_ADAP_EM_PATH', 'embeddings/domain_adapted/')

PUNCTUATION = r"""!"#%&'()*+,-./:;<=>?@[\]^_`{|}~‘’"""

general_embeddings_vocab = os.environ.get('GEN_VOCAB', 'embeddings/general/')
general_embeddings_vectors = os.environ.get('GEN_VECTOR', 'embeddings/general/')