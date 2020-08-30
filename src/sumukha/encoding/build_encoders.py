import numpy as np
from brushes.io import filesystem
from sumukha.config import input_preprocess_path


def general_encoders(preprocess_path, embeddings_path):
    """
    Uses general Fasttext embeddings to encode the data set.
    :param preprocess_path:
    :param embeddings_path:
    :return:
    """
    dataset = filesystem.load_obj(input_preprocess_path, 'dataset')['features']

    # general purpose
    vocab = filesystem.load_obj(embeddings_path, 'vocab')
    vectors = filesystem.load_obj(embeddings_path, 'vectors')

    word_set = {word for sentence in dataset['review'] for word in sentence.split()}
    # refined vocabulary
    vocab = {key: value for key, value in vocab.items() if key in word_set or int(value) < 50000}

    vectors = vectors[list(vocab.values())]
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)
    # stack a random vector for unk token
    vectors = np.vstack((vectors, np.random.rand(vectors.shape[1])))
    vocab = {key: index for index, key in enumerate(vocab.keys())}

    filesystem.save_obj(vocab, preprocess_path, 'general_vocab_processed')
    filesystem.save_obj(vectors, preprocess_path, 'general_vector_processed')


def domain_encoders(preprocess_path, model_result_path):
    """

    :param dataset:
    :return:
    """
    pass
