import os

from sumukha import config


def run_upload(root_path='./', input_file_path=config.input_file_path + 'imdb_dataset.csv'):
    """
    Loads the file form the input path and places it inside the project
    :param root_path: <str>
    :param input_file_path: <str>
    :return: None
    """
    result = config.input_file_path
    os.system(f'cp {input_file_path} {result}')
    print('Upload Successful')


if __name__ == '__main__':
    run_upload()
