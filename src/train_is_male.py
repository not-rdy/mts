import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import save_f, load_f
from base.settings import PATH_DATA_INTERIM
from lib.params_is_male import params, params_model
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn.aggr import MaxAggregation
from sklearn.metrics import mean_squared_error

PATH_GRAPHS = os.path.join(PATH_DATA_INTERIM, 'is_male')

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

device = torch.device(params['device'])
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
    batch_size=params['batch_size'],
    shuffle=True)
val = DataLoader(
    dataset=val,
    batch_size=params['batch_size'],
    shuffle=False)
test = DataLoader(
    dataset=test,
    batch_size=params['batch_size'],
    shuffle=False)

model = GraphSAGE(**params_model).to(device)
agg_fun = MaxAggregation()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params['lr'],
    weight_decay=params['weight_decay'])
loss_fun = torch.nn.MSELoss()


if __name__ == '__main__':

    mlflow.set_experiment('GNN_is_male')
    mlflow.start_run()

    list_out_test = []
    list_y_test = []
    for epoch in range(1, params['n_epochs']+1):

        list_out_train = []
        list_y_train = []

        list_out_val = []
        list_y_val = []

        print(f"[Epoch: {epoch}]")

        model.train()
        for batch in tqdm(
                train, total=len(train), colour='green'):

            out = model(batch.x, batch.edge_index)
            out = agg_fun(out, batch.batch)
            out = torch.sigmoid(out).reshape(-1)
            y = batch.y.type(torch.cuda.FloatTensor)
            loss = loss_fun(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out = [x.item() for x in out]
            y = list(y.cpu().numpy())

            list_out_train.extend(out)
            list_y_train.extend(y)

        rmse_train = mean_squared_error(
            y_true=list_y_train, y_pred=list_out_train, squared=False)
        mlflow.log_metric(
            key='rmse_train', value=rmse_train, step=epoch)

        save_f(
            filename=os.path.join(PATH_DATA_INTERIM, 'model.pkl'),
            obj=model)
        mlflow.log_artifact(
            os.path.join(PATH_DATA_INTERIM, f'model_{epoch}.pkl'))

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                    val, total=len(val), colour='green'):

                out = model(batch.x, batch.edge_index)
                out = agg_fun(out, batch.batch)
                out = torch.sigmoid(out)
                y = batch.y

                out = [x.item() for x in out]
                y = list(y.cpu().numpy())

                list_out_val.extend(out)
                list_y_val.extend(y)

        rmse_val = mean_squared_error(
            y_true=list_y_val, y_pred=list_out_val, squared=False)
        mlflow.log_metric(
            key='rmse_val', value=rmse_val, step=epoch)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
                test, total=len(test), colour='green'):

            out = model(batch.x, batch.edge_index)
            out = agg_fun(out, batch.batch)
            out = torch.sigmoid(out)
            y = batch.y

            out = [x.item() for x in out]
            y = list(y.cpu().numpy())

            list_out_test.extend(out)
            list_y_test.extend(y)

    rmse_test = mean_squared_error(
        y_true=list_y_test, y_pred=list_out_test, squared=False)
    mlflow.log_metric(
        key='rmse_test', value=rmse_test, step=epoch)

    mlflow.log_params(params=params)

    mlflow.end_run()
