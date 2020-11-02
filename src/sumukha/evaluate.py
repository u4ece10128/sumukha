from sumukha.config import input_preprocess_path
from sumukha.evaluation.evaluate_embeddings import Evaluate


def run_evaluate():
    """
    Evaluate the test set. Publish results on performances on all the embedders
    :return:
    """
    evaluate = Evaluate(embeddings_path=input_preprocess_path)
    evaluate.evaluate_general_embeddings()
    evaluate.evaluate_domain_embeddings()
    evaluate.evaluate_domain_adapted_embeddings()


if __name__ == '__main__':
    run_evaluate()
