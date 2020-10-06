import fasttext as ft

import numpy as np

from brushes.io import filesystem

from sklearn.cross_decomposition import CCA

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


def domain_encoders(preprocess_path, trained_embeddings_path):
    """
    Uses fasttext embeddings trained on the domain data set.
    :param preprocess_path:
    :param trained_embeddings_path
    :return:
    """
    model = ft.load_model(trained_embeddings_path + 'vectors.bin')
    vocab = model.words

    vocab = {key: index for index, key in enumerate(vocab)}

    vocab = {key: index for index, key in enumerate(vocab)}
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)

    vectors = np.zeros((len(vocab), 100), dtype=np.float32)
    for word, index in vocab.items():
        vectors[index] = model.get_word_vector(word)

    filesystem.save_obj(vocab, preprocess_path, 'domain_vocab_processed')
    filesystem.save_obj(vectors, preprocess_path, 'domain_vector_processed')


def mixing_embeddings(preprocess_path, embeddings_path_domain_adapted):
    """
    This method performs Canonical Correlational Analysis on the common
    vocabulary words
    :param preprocess_path:
    :param embeddings_path_domain_adapted
    :return:
    """
    # Read Vocabulary.
    general_vocabs = filesystem.load_obj(preprocess_path, 'general_vocab_processed')
    general_vectors = filesystem.load_obj(preprocess_path, 'general_vector_processed')

    domain_vocabs = filesystem.load_obj(preprocess_path, 'domain_vocab_processed')
    domain_vectors = filesystem.load_obj(preprocess_path, 'domain_vector_processed')

    intersection_vocab = dict()
    for vocab in general_vocabs:
        if vocab in domain_vocabs:
            intersection_vocab[vocab] = [general_vocabs[vocab], domain_vocabs[vocab]]

    print('Number of General Related Vocabularies: ', len(general_vocabs))
    print('Number of Domain Related Vocabularies: ', len(domain_vocabs))
    print('Number of Common Vocabularies: ', len(intersection_vocab))

    mixed_embeddings = {}

    for common_vocab in intersection_vocab:
        vecotr_g = general_vectors[general_vocabs[common_vocab]]
        vector_ds = domain_vectors[domain_vocabs[common_vocab]]

        cca = CCA(n_components=1)
        cca.fit(vecotr_g.reshape(-1, 1), vector_ds.reshape(-1, 1))
        projection_g, projection_ds = cca.transform(vecotr_g.reshape(-1, 1), vector_ds.reshape(-1, 1))

        # alpha and beta can be tuned
        alpha = 0.5
        beta = 0.5

        wg_hat = vecotr_g * projection_g.T
        wds_hat = vector_ds * projection_ds.T
        wda = (alpha * wg_hat) + (beta * wds_hat)
        mixed_embeddings[common_vocab] = wda
    intersection_vocab = {key: index for index, key in enumerate(intersection_vocab.keys())}
    vectors = np.zeros((len(intersection_vocab), 100), dtype=np.float32)

    for index, word in enumerate(intersection_vocab.keys()):
        vectors[index] = mixed_embeddings.get(word)[0]

    filesystem.save_obj(intersection_vocab, preprocess_path, 'adapted_vocab_processed')
    filesystem.save_obj(vectors, preprocess_path, 'adapted_vector_processed')
