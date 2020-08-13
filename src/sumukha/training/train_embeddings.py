from brushes.io.filesystem import save_obj
from sumukha.config import input_preprocess_path, embeddings_path_domain
import fasttext


def train_embeddings(preprocess_path=input_preprocess_path, model_result_path=embeddings_path_domain):
    model = fasttext.train_unsupervised(preprocess_path + 'data.txt', model='skipgram', verbose=2)
    vocab = model.words
    model.save_model(model_result_path + "vectors.bin")

    save_obj(vocab, embeddings_path_domain, 'vocab')
