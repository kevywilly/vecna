import os

ROOT_DIR = os.path.abspath(os.curdir)
MODEL_FOLDER = f'{ROOT_DIR}/model'


def model_path(filename: str):
    return os.path.join(MODEL_FOLDER, filename)