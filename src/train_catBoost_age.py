import os
import mlflow
import torch
import numpy as np
from tqdm import tqdm
from lib.utils import load_f
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import MaxAggregation
from catboost import CatBoostClassifier
from base.settings import PATH_DATA_INTERIM
from sklearn.metrics import f1_score

device = torch.device('cuda')
PATH_GRAPHS = os.path.join(PATH_DATA_INTERIM, 'age')

f_names = os.listdir(PATH_GRAPHS)
train_names = [x for x in f_names if 'train' in x]
val_names = [x for x in f_names if 'val' in x]
test_names = [x for x in f_names if 'test' in x]

train_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in train_names]
val_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in val_names]
test_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in test_names]

train = []
for part in train_parts:
    part = [x.to(device) for x in part]
    train.extend(part)
del train_parts
val = []
for part in val_parts:
    part = [x.to(device) for x in part]
    val.extend(part)
del val_parts
test = []
for part in test_parts:
    part = [x.to(device) for x in part]
    test.extend(part)
del test_parts

print(f"n graphs train {len(train)}")
print(f"n graphs val {len(val)}")
print(f"n graphs test {len(test)}")

train = DataLoader(
    dataset=train,
    batch_size=16,
    shuffle=True)
val = DataLoader(
    dataset=val,
    batch_size=16,
    shuffle=False)
test = DataLoader(
    dataset=test,
    batch_size=16,
    shuffle=False)

path_GNNModel_age = mlflow.artifacts.download_artifacts(
    artifact_uri='runs:/bdfe7e4b51464ee1855eae1afe6b70b1/model_3.pkl')
GNNModel_age = load_f(path_GNNModel_age).to(device)
agg_fun = MaxAggregation()

print('Getting train features ...')
list_train_features = []
list_train_y = []
with torch.no_grad():
    for batch in tqdm(train, total=len(train), colour='green'):

        train_features = GNNModel_age(batch.x, batch.edge_index)
        train_features = agg_fun(train_features, batch.batch)

        train_features = train_features.cpu().numpy()
        train_y = batch.y.cpu().numpy()

        list_train_features.append(train_features)
        list_train_y.append(train_y)
train_features = np.concatenate(list_train_features, axis=0)
train_y = np.concatenate(list_train_y, axis=0)

print('Getting val features ...')
list_val_features = []
list_val_y = []
with torch.no_grad():
    for batch in tqdm(val, total=len(val), colour='green'):

        val_features = GNNModel_age(batch.x, batch.edge_index)
        val_features = agg_fun(val_features, batch.batch)

        val_features = val_features.cpu().numpy()
        val_y = batch.y.cpu().numpy()

        list_val_features.append(val_features)
        list_val_y.append(val_y)
val_features = np.concatenate(list_val_features, axis=0)
val_y = np.concatenate(list_val_y, axis=0)

print('Getting test features ...')
list_test_features = []
list_test_y = []
with torch.no_grad():
    for batch in tqdm(test, total=len(test), colour='green'):

        test_features = GNNModel_age(batch.x, batch.edge_index)
        test_features = agg_fun(test_features, batch.batch)

        test_features = test_features.cpu().numpy()
        test_y = batch.y.cpu().numpy()

        list_test_features.append(test_features)
        list_test_y.append(test_y)
test_features = np.concatenate(list_test_features, axis=0)
test_y = np.concatenate(list_test_y, axis=0)


print('Train CatBoost ...')
cat = CatBoostClassifier(
    iterations=10000,
    early_stopping_rounds=1000,
    use_best_model=True,
    verbose=100,
    task_type='GPU',
    devices='0')
cat.fit(
    X=train_features, y=train_y,
    eval_set=(val_features, val_y))

predict_y_train = cat.predict(train_features)
predict_y_val = cat.predict(val_features)
predict_y_test = cat.predict(test_features)

f1_train_micro = f1_score(
    y_true=train_y,
    y_pred=predict_y_train,
    average='micro')
f1_train_macro = f1_score(
    y_true=train_y,
    y_pred=predict_y_train,
    average='macro')

f1_val_micro = f1_score(
    y_true=val_y,
    y_pred=predict_y_val,
    average='micro')
f1_val_macro = f1_score(
    y_true=val_y,
    y_pred=predict_y_val,
    average='macro')

f1_test_micro = f1_score(
    y_true=test_y,
    y_pred=predict_y_test,
    average='micro')
f1_test_macro = f1_score(
    y_true=test_y,
    y_pred=predict_y_test,
    average='macro')

print(f"f1_train_micro: {f1_train_micro}")
print(f"f1_train_macro: {f1_train_macro}")
print(f"f1_val_micro: {f1_val_micro}")
print(f"f1_val_macro: {f1_val_macro}")
print(f"f1_test_micro: {f1_test_micro}")
print(f"f1_test_macro: {f1_test_macro}")
