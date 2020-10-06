from brushes.io.filesystem import save_obj
from sumukha.config import input_preprocess_path, embeddings_path_domain
import fasttext


def train_embeddings(preprocess_path=input_preprocess_path, model_result_path=embeddings_path_domain):
    model = fasttext.train_supervised(preprocess_path + 'data_train.txt', lr=1.0, wordNgrams=2, epoch=20,
                                      verbose=2)
    number_samples, precision, recall = model.test(preprocess_path + 'data_test.txt')

    print('Number of samples to test: {}'.format(number_samples))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))

    vocab = model.words
    model.save_model(model_result_path + "vectors.bin")

    save_obj(vocab, embeddings_path_domain, 'vocab')
