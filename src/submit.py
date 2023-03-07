import os
import torch
import mlflow
import pandas as pd
from tqdm import tqdm
from lib.utils import load_f
from torch_geometric.loader import DataLoader
from base.settings import PATH_DATA_INTERIM

PATH_GRAPHS = os.path.join(PATH_DATA_INTERIM, 'none')
f_names = os.listdir(PATH_GRAPHS)
submit_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in f_names]

device = torch.device('cuda')
submit = []
for part in submit_parts:
    part = [x.to(device) for x in part]
    submit.extend(part)
del submit_parts

print(f"n graphs submit {len(submit)}")

submit = DataLoader(
    dataset=submit,
    batch_size=16,
    shuffle=False)

path_model_age = mlflow.artifacts.download_artifacts(
    artifact_uri='runs:/0a783b32c4134355aa56f4dc0bde9a16/model.pkl')
path_agg_fun = mlflow.artifacts.download_artifacts(
    artifact_uri='runs:/0a783b32c4134355aa56f4dc0bde9a16/agg_fun.pkl')
model_age = load_f(path_model_age).to(device)
agg_fun = load_f(path_agg_fun).to(device)

list_user_id = []
list_out_age = []
for batch in tqdm(submit, total=len(submit), colour='green'):

    users_id = [x.item() for x in batch.user_id]
    out_age = model_age(batch.x, batch.edge_index)
    out_age = agg_fun(out_age, batch.batch)
    out_age = torch.softmax(out_age, dim=0)
    out_age = list(torch.argmax(out_age, dim=1).cpu().numpy())
    out_age = [x + 1 for x in out_age]

    list_user_id.extend(users_id)
    list_out_age.extend(out_age)

submit = pd.DataFrame({
    'user_id': list_user_id,
    'age': list_out_age,
    'is_male': [None] * len(list_out_age)})
submit.to_csv(
    os.path.join(PATH_DATA_INTERIM, 'submission.csv'))
