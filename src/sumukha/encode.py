from sumukha import config
from sumukha.encoding.build_encoders import general_encoders, domain_encoders, mixing_embeddings


def run_encode(root_path='./', preprocess_path=config.input_preprocess_path,
               general_embeddings_path=config.embeddings_path_general, model_result_path=config.embeddings_path_domain):
    general_encoders(preprocess_path=root_path + preprocess_path, embeddings_path=root_path + general_embeddings_path)
    domain_encoders(preprocess_path=root_path + preprocess_path,
                    trained_embeddings_path=root_path + model_result_path)

    mixing_embeddings(preprocess_path=root_path + preprocess_path,
                      embeddings_path_domain_adapted=root_path + config.embeddings_path_domain_adapted)


if __name__ == '__main__':
    run_encode()
