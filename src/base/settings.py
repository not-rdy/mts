import os
import sqlite3

PATH_PROJ = '/home/rustem/projects/kaggle_mts'
PATH_DATA = os.path.join(PATH_PROJ, 'data')
PATH_DATA_RAW = os.path.join(PATH_DATA, 'raw')
PATH_DATA_INTERIM = os.path.join(PATH_DATA, 'interim')
PATH_DATABASE = os.path.join(PATH_DATA, 'db')
PATH_MODELS = os.path.join(PATH_PROJ, 'models')

connector = sqlite3.connect(
    os.path.join(PATH_DATABASE, 'mts.db'))
