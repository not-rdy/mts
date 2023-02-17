import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils.convert import from_networkx
from lib.utils import save_f
from lib.DBManager import DBManager
from base.settings import connector, PATH_DATA_INTERIM


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
            agg_data
        where
            user_id in ({', '.join(users_part)})
        """
    )
    users_part['date'] = pd.to_datetime(users_part['date'], format='%Y-%m-%d')
    users_part = users_part.sort_values(
        by=['user_id', 'date', 'part_of_day_encoded'])
    users_part['day'] = users_part['date'].dt.day
    users_part['month'] = users_part['date'].dt.month
    users_part['year'] = users_part['date'].dt.year
    users_part['day_of_week'] = users_part['date'].dt.day_of_week
    users_part['day_of_year'] = users_part['date'].dt.day_of_year
    users_part = pd.get_dummies(data=users_part, columns=['part_of_day'])
    users_part = users_part.set_index(['user_id'])
    users_part = users_part.drop(['date', 'part_of_day_encoded'], axis=1)

    list_users_graph = []
    for idx in users_part.index.unique():
        user_seq = users_part.loc[idx]
        if type(user_seq) == pd.DataFrame:
            user_graph = nx.DiGraph(y=[user_seq.iloc[0]['age']])
            user_seq = list(user_seq.itertuples(index=False, name=None))
        else:
            user_graph = nx.DiGraph(y=[user_seq['age'].item()])
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
            PATH_DATA_INTERIM,
            f'users_graph_{first_user}_{last_user}'),
        obj=list_users_graph)


db = DBManager(connector)

users_all = db.read_data_as_df(
    """
    select
        distinct user_id
    from
        agg_data
    order by
        random()
    """
)
users_all = users_all['user_id'].tolist()

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
