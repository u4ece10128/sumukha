import click

from sumukha.config import input_preprocess_path
from sumukha.preprocessor import run_preprocess
from sumukha.upload import run_upload
from sumukha.train import run_train
from sumukha.encode import run_encode
from sumukha.evaluate import run_evaluate


@click.group()
@click.option('--root', '-r', 'root_path', required=True, type=str, envvar='ROOT_PROJECT_PATH',
              help='specify the root path')
@click.pass_context
def cli(ctx, root_path):
    """
    Welcome to sumukha the data adaptive encoder
    """
    ctx.obj = root_path


@cli.command()
@click.option('--upload', '-u', 'input_', default=input_preprocess_path, show_default=True, type=str)
def upload(input_):
    """
    Uploads the data set onto sumukha
    """
    run_upload(input_file_path=input_)
    run_preprocess()


@cli.command()
def train():
    """
    Train encoders on the uploaded data set
    :return:
    """
    run_train()
    run_encode()


@cli.command()
def evaluate():
    """
    Evaluate the embeddings
    :return:
    """
    run_evaluate()


if __name__ == '__main__':
    cli(obj='')
