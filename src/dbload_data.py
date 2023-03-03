# %%
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from lib.DBManager import DBManager
from base.settings import connector, PATH_DATA_RAW

db = DBManager(conn=connector)

filenames = os.listdir(
    os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt'))
filenames = [x for x in filenames if x != '_SUCCESS']

target = pq.read_table(
     os.path.join(PATH_DATA_RAW, 'public_train.pqt')).to_pandas()
target = target.set_index('user_id')

id_users_all = []
for name in filenames:
    id_users = pq.read_table(
        os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt', name),
        columns=['user_id']).to_pandas()['user_id'].unique()
    id_users = list(id_users)
    id_users_all.extend(id_users)
id_users_all = pd.Series(np.unique(id_users_all))

id_users_submit = pq.read_table(os.path.join(PATH_DATA_RAW, 'submit_2.pqt'))\
    .to_pandas()['user_id'].tolist()

np.random.seed(1411)
id_users_train = id_users_all[
    ~id_users_all.isin(id_users_submit)].sample(frac=0.6).tolist()
id_users_val = id_users_all[
    (~id_users_all.isin(id_users_submit))
    & (~id_users_all.isin(id_users_train))].sample(frac=0.5).tolist()
id_users_test = id_users_all[
    (~id_users_all.isin(id_users_submit))
    & (~id_users_all.isin(id_users_train))
    & (~id_users_all.isin(id_users_val))].tolist()
del id_users_all

for name in filenames:
    print(f'filename: {name}')
    region_names = pq.read_table(
        os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt', name),
        columns=['region_name']).to_pandas()['region_name'].unique()

    for region in tqdm(region_names, total=len(region_names)):
        part = pq.read_table(
            os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt', name),
            filters=[('region_name', '=', region)]).to_pandas()
        part_submit = part[part['user_id'].isin(id_users_submit)].copy()
        part_train = part[part['user_id'].isin(id_users_train)].copy()
        part_val = part[part['user_id'].isin(id_users_val)].copy()
        part_test = part[part['user_id'].isin(id_users_test)].copy()
        part_train['age'] = part_train['user_id'].map(target['age'])
        part_val['age'] = part_val['user_id'].map(target['age'])
        part_test['age'] = part_test['user_id'].map(target['age'])
        del part

        if part_submit.shape[0] > 0:
            db.write_df(
                df=part_submit,
                table_name='data_submit',
                if_exist='append')
        if part_train.shape[0] > 0:
            db.write_df(
                df=part_train,
                table_name='data_train',
                if_exist='append')
        if part_val.shape[0] > 0:
            db.write_df(
                df=part_val,
                table_name='data_val',
                if_exist='append')
        if part_test.shape[0] > 0:
            db.write_df(
                df=part_test,
                table_name='data_test',
                if_exist='append')
