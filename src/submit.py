import os
import torch
import mlflow
import pandas as pd
from tqdm import tqdm
from lib.utils import load_f
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import MaxAggregation
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
    artifact_uri='runs:/39df34b9de404518afa4f7e25b1d3820/model.pkl')
path_model_is_male = mlflow.artifacts.download_artifacts(
    artifact_uri='runs:/2ebc84bc596d4027925fa916214199f7/model_1000.pkl'
)

model_age = load_f(path_model_age).to(device)
model_is_male = load_f(path_model_is_male).to(device)
agg_fun = MaxAggregation()

list_user_id = []
list_out_age = []
list_out_is_male = []
for batch in tqdm(submit, total=len(submit), colour='green'):

    users_id = [x.item() for x in batch.user_id]

    out_age = model_age(batch.x, batch.edge_index)
    out_age = agg_fun(out_age, batch.batch)
    out_age = torch.softmax(out_age, dim=0)
    out_age = list(torch.argmax(out_age, dim=1).cpu().numpy())
    out_age = [x + 1 for x in out_age]

    out_is_male = model_is_male(batch.x, batch.edge_index)
    out_is_male = agg_fun(out_is_male, batch.batch)
    out_is_male = torch.sigmoid(out_is_male)
    out_is_male = [x.item() for x in out_is_male]

    list_user_id.extend(users_id)
    list_out_age.extend(out_age)
    list_out_is_male.extend(out_is_male)

submit = pd.DataFrame({
    'user_id': list_user_id,
    'age': list_out_age,
    'is_male': list_out_is_male
    })
submit.to_csv(
    os.path.join(PATH_DATA_INTERIM, 'submission.csv'))
