import os
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils.convert import from_networkx
from lib.utils import save_f
from lib.DBManager import DBManager
from base.settings import connector, PATH_DATA_INTERIM, PATH_DATA_RAW


def get_age_bucket(x: int) -> int:
    # age buckets
    b1 = pd.Interval(left=18, right=25, closed='right')
    b2 = pd.Interval(left=25, right=35, closed='right')
    b3 = pd.Interval(left=35, right=45, closed='right')
    b4 = pd.Interval(left=45, right=55, closed='right')
    b5 = pd.Interval(left=55, right=65, closed='right')
    b6 = pd.Interval(left=65, right=np.inf, closed='right')
    if x in b1:
        return 0
    elif x in b2:
        return 1
    elif x in b3:
        return 2
    elif x in b4:
        return 3
    elif x in b5:
        return 4
    elif x in b6:
        return 5
    else:
        return np.nan


def create_graphs(users_part: list) -> list:

    first_user, last_user = users_part[0], users_part[-1]

    users_part = [str(x) for x in users_part]
    users_part = db.read_data_as_df(
        f"""
        select
            *,
            case part_of_day
                when 'morning' then 0
                when 'day' then 1
                when 'evening' then 2
                when 'night' then 3
            end part_of_day_encoded
        from
            {data_type}_agg
        where
            user_id in ({', '.join(users_part)})
        """
    )
    numeric_cols = [
        'region_name', 'city_name', 'cpe_manufacturer_name',
        'cpe_model_name', 'url_host', 'cpe_type_cd', 'cpe_model_os_type',
        'price', 'request_cnt']
    users_part.loc[:, numeric_cols] = users_part.loc[:, numeric_cols]\
        .fillna(users_part.loc[:, numeric_cols].mean())
    users_part['date'] = pd.to_datetime(users_part['date'], format='%Y-%m-%d')
    users_part = users_part.sort_values(
        by=['user_id', 'date', 'part_of_day_encoded'])
    users_part['day'] = users_part['date'].dt.day
    users_part['month'] = users_part['date'].dt.month
    users_part['year'] = users_part['date'].dt.year
    users_part['day_of_week'] = users_part['date'].dt.day_of_week
    users_part['day_of_year'] = users_part['date'].dt.day_of_year
    users_part = pd.get_dummies(data=users_part, columns=['part_of_day'])
    users_part = users_part.drop(['date', 'part_of_day_encoded'], axis=1)

    users_part['age'] = users_part['user_id'].map(mapper_target['age'])
    users_part['age'] = users_part['user_id'].map(lambda x: get_age_bucket(x))
    users_part['is_male'] = users_part['user_id'].map(mapper_target['is_male'])
    users_part = users_part.set_index(['user_id'])
    if target == 'age':
        users_part = users_part.drop('is_male', axis=1)
    elif target == 'is_male':
        users_part = users_part.drop('age', axis=1)
        users_part = users_part[users_part['is_male'] != 'NA']
        users_part = users_part[
            users_part['is_male'].map(lambda x: not pd.isna(x))]
        users_part['is_male'] = users_part['is_male'].astype(int)

    cols_date = [
        'day', 'month', 'year',
        'day_of_week', 'day_of_year']
    for col in cols_date:
        x = users_part[col].copy()
        x_std = (x - x.min()) / (x.max() - x.min())
        users_part[col] = x_std.tolist()
    cols_other = [
        'region_name', 'city_name', 'cpe_manufacturer_name',
        'cpe_model_name', 'url_host', 'cpe_type_cd',
        'cpe_model_os_type', 'price', 'request_cnt']
    stats = db.read_data_as_df(
        f"""
        select
            min(region_name) as min_region_name,
            min(city_name) as min_city_name,
            min(cpe_manufacturer_name) as min_cpe_manufacturer_name,
            min(cpe_model_name) as min_cpe_model_name,
            min(url_host) as min_url_host,
            min(cpe_type_cd) as min_cpe_type_cd,
            min(cpe_model_os_type) as min_cpe_model_os_type,
            min(price) as min_price,
            min(request_cnt) as min_request_cnt,
            max(region_name) as max_region_name,
            max(city_name) as max_city_name,
            max(cpe_manufacturer_name) as max_cpe_manufacturer_name,
            max(cpe_model_name) as max_cpe_model_name,
            max(url_host) as max_url_host,
            max(cpe_type_cd) as max_cpe_type_cd,
            max(cpe_model_os_type) as max_cpe_model_os_type,
            max(price) as max_price,
            max(request_cnt) as max_request_cnt
        from
            {data_type}_agg
        """
    )
    for col in cols_other:
        x = users_part[col].copy()
        if col == 'region_name':
            col_min = 'min_region_name'
            col_max = 'max_region_name'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'city_name':
            col_min = 'min_city_name'
            col_max = 'max_city_name'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'cpe_manufacturer_name':
            col_min = 'min_cpe_manufacturer_name'
            col_max = 'max_cpe_manufacturer_name'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'cpe_model_name':
            col_min = 'min_cpe_model_name'
            col_max = 'max_cpe_model_name'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'url_host':
            col_min = 'min_url_host'
            col_max = 'max_url_host'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'cpe_type_cd':
            col_min = 'min_cpe_type_cd'
            col_max = 'max_cpe_type_cd'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'cpe_model_os_type':
            col_min = 'min_cpe_model_os_type'
            col_max = 'max_cpe_model_os_type'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'price':
            col_min = 'min_price'
            col_max = 'max_price'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        elif col == 'request_cnt':
            col_min = 'min_request_cnt'
            col_max = 'max_request_cnt'
            x_std = (x - stats[col_min].item()) /\
                (stats[col_max].item() - stats[col_min].item())
        users_part[col] = x_std.tolist()

    list_users_graph = []
    for idx in users_part.index.unique():
        user_seq = users_part.loc[idx]
        if type(user_seq) == pd.DataFrame:
            user_graph = nx.DiGraph(y=[user_seq.iloc[0][target]])
            user_seq = list(user_seq.itertuples(index=False, name=None))
        else:
            if target == 'age':
                user_graph = nx.DiGraph(y=[user_seq[target].item()])
            elif target == 'is_male':
                user_graph = nx.DiGraph(y=[user_seq[target]])
            user_seq = [tuple(user_seq)]

        list_nodes = []
        for idx, node in enumerate(user_seq):
            list_nodes.append(
                (idx, {'x': node}))
        del user_seq

        list_edges = []
        indices = list(range(0, len(list_nodes)))
        for idx_l, idx_r in zip(indices[:-1], indices[1:]):
            list_edges.append(
                (idx_l, idx_r))
        del indices

        user_graph.add_nodes_from(list_nodes)
        user_graph.add_edges_from(list_edges)
        user_graph = from_networkx(user_graph)

        list_users_graph.append(user_graph)
        del user_graph

    save_f(
        filename=os.path.join(
            PATH_DATA_INTERIM, target,
            f'{data_type}_users_graph_{first_user}_{last_user}'),
        obj=list_users_graph)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-target',
        type=str)
    parser.add_argument(
        '-data_type',
        type=str)
    args = parser.parse_args()

    target = args.target
    data_type = args.data_type

    db = DBManager(connector)

    users_all = db.read_data_as_df(
        f"""
        select
            distinct user_id
        from
            {data_type}_agg
        """
    )
    users_all = users_all['user_id'].tolist()

    mapper_target = pq.read_table(
        os.path.join(PATH_DATA_RAW, 'public_train.pqt')).to_pandas()
    mapper_target = mapper_target.set_index('user_id')

    step = 10000
    indices = list(range(0, len(users_all), step))
    indices.append(len(users_all))
    list_users_part = []
    for idx_left, idx_right in zip(indices[:-1], indices[1:]):
        users_part = users_all[idx_left:idx_right]
        list_users_part.append(users_part)

    parts_to_dataloaders = []
    with Pool(8) as p:
        for res in tqdm(
                p.imap_unordered(create_graphs, list_users_part),
                total=len(list_users_part)):
            parts_to_dataloaders.append(res)
