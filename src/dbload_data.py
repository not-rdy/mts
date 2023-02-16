import os
import pyarrow.parquet as pq
from tqdm import tqdm
from lib.DBManager import DBManager
from base.settings import connector, PATH_DATA_RAW

db = DBManager(conn=connector)

target = pq.read_table(
    os.path.join(PATH_DATA_RAW, 'public_train.pqt')).to_pandas()

filenames = os.listdir(
    os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt'))
filenames = [x for x in filenames if x != '_SUCCESS']

nrows = 0
for name in filenames:
    print(f'filename: {name}')
    region_names = pq.read_table(
        os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt', name),
        columns=['region_name']).to_pandas()['region_name'].unique()

    for region in tqdm(region_names, total=len(region_names)):
        part = pq.read_table(
            os.path.join(PATH_DATA_RAW, 'competition_data_final_pqt', name),
            filters=[('region_name', '=', region)]).to_pandas()
        part['age'] = part['user_id'].map(target.set_index('user_id')['age'])

        nrows += part.shape[0]

        db.write_df(
            df=part,
            table_name='data',
            if_exist='append')

    print(f"rows in db: {nrows}")
