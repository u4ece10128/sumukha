import numpy as np

from brushes.io.filesystem import load_obj, save_obj

from sumukha.config import input_preprocess_path

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


class Evaluate:
    def __init__(self, embeddings_path):
        self.dataset = load_obj(input_preprocess_path, 'dataset')['features']
        self.embeddings_path = embeddings_path

    @staticmethod
    def _encode_sentences(sentence, vectors, vocab):
        """

        :param sentence:
        :param vectors:
        :param vocab:
        :return:
        """
        result = np.zeros((1, vectors.shape[1]), dtype=np.float32)
        count = 0
        for token in sentence.split(' '):
            if token in vocab:
                result += vectors[vocab[token]]
                count += 1
            else:
                result += vectors[vocab['<unk>']]
                count += 1
        return result

    def _prepare_dataset(self, vectors, vocab):
        """

        :return:
        """
        reviews = self.dataset['review'].copy()
        labels = self.dataset['sentiment'].copy()

        reviews = reviews.apply(lambda x: self._encode_sentences(x, vectors, vocab))
        reviews = reviews.apply(lambda x: x.reshape(100, ))
        reviews = np.array([samples for samples in reviews])

        labels = np.array([lab for lab in labels])
        labels = labels.reshape(len(labels), 1)

        train_x, test_x, train_y, test_y = train_test_split(reviews, labels, test_size=0.3)
        return train_x, train_y, test_x, test_y

    def _model_tf(self, x_train, y_train, x_test, y_test):
        """

        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """

        model = Sequential()
        model.add(Input(shape=(100,)))
        model.add(Dense(128, activation='linear', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='linear', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='softmax', kernel_initializer='he_uniform'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

        model.fit(x_train, to_categorical(y_train), batch_size=32, epochs=20, validation_split=0.1, verbose=2)

        test_results = model.evaluate(x_test, to_categorical(y_test), verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1] * 100}%')

    def evaluate_general_embeddings(self):
        """

        :return:
        """
        vocab = load_obj(self.embeddings_path, 'general_vocab_processed')
        vectors = load_obj(self.embeddings_path, 'general_vector_processed')

        x_train, y_train, x_test, y_test = self._prepare_dataset(vectors, vocab)

        self._model_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def evaluate_domain_embeddings(self):
        """

        :return:
        """
        vocab = load_obj(self.embeddings_path, 'domain_vocab_processed')
        vectors = load_obj(self.embeddings_path, 'domain_vector_processed')

        x_train, y_train, x_test, y_test = self._prepare_dataset(vectors, vocab)

        self._model_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def evaluate_domain_adapted_embeddings(self):
        """

        :return:
        """
        vocab = load_obj(self.embeddings_path, 'adapted_vocab_processed')
        vectors = load_obj(self.embeddings_path, 'adapted_vector_processed')

        x_train, y_train, x_test, y_test = self._prepare_dataset(vectors, vocab)

        self._model_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
