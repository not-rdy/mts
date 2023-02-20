import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import load_f
from base.settings import PATH_DATA_INTERIM
from lib.models import GCN, params_GCN
from torch_geometric.loader import DataLoader

f_names = os.listdir(PATH_DATA_INTERIM)
f_names = [x for x in f_names if 'users' in x]

train_names = f_names[:20]
val_names = f_names[20:25]
test_names = f_names[25:]

train_parts = [
    load_f(os.path.join(PATH_DATA_INTERIM, name)) for name in train_names]
val_parts = [
    load_f(os.path.join(PATH_DATA_INTERIM, name)) for name in val_names]
test_parts = [
    load_f(os.path.join(PATH_DATA_INTERIM, name)) for name in test_names]

device = torch.device(params_GCN['device'])
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

batch_size = params_GCN['batch_size']
train = DataLoader(train, batch_size)
val = DataLoader(val, batch_size)
test = DataLoader(test, batch_size)

model = GCN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params_GCN['lr'],
    weight_decay=params_GCN['weight_decay'])
loss_fun = torch.nn.CrossEntropyLoss()


if __name__ == '__main__':

    mlflow.set_experiment('GCN')
    mlflow.start_run()

    for epoch in range(1, params_GCN['n_epochs']+1):

        list_loss_train = []
        list_loss_val = []
        print(f"[Epoch: {epoch}]")

        model.train()
        for batch in tqdm(
                train, total=len(train), colour='green'):

            out = model(batch)
            y = batch.y.type(torch.cuda.ByteTensor)
            loss = loss_fun(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            list_loss_train.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                    val, total=len(val), colour='green'):

                out = model(batch)
                y = batch.y.type(torch.cuda.ByteTensor)
                loss = loss_fun(out, y)

                list_loss_val.append(loss.item())

        loss_train = sum(list_loss_train) / len(list_loss_train)
        mlflow.log_metric(key='loss train', value=loss_train, step=epoch)
        loss_val = sum(list_loss_val) / len(list_loss_val)
        mlflow.log_metric(key='loss val', value=loss_val, step=epoch)

        mlflow.log_params(params=params_GCN)

    mlflow.end_run()
