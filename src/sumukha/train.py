from sumukha import config
from sumukha.training.train_embeddings import train_embeddings


def run_train(root_path='./', preprocess_path=config.input_preprocess_path,
              model_result_path=config.embeddings_path_domain):
    train_embeddings(preprocess_path=root_path + preprocess_path, model_result_path=root_path + model_result_path)


if __name__ == '__main__':
    run_train()
