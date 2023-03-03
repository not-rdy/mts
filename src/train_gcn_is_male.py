import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import save_f, load_f
from base.settings import PATH_DATA_INTERIM
from lib.models import GCN_is_male, params_GCN
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

PATH_GRAPHS = os.path.join(PATH_DATA_INTERIM, 'is_male')

f_names = os.listdir(PATH_GRAPHS)
f_names = [x for x in f_names if 'users' in x]

train_names = f_names[:20]
val_names = f_names[20:25]
test_names = f_names[25:]

train_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in train_names]
val_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in val_names]
test_parts = [
    load_f(os.path.join(PATH_GRAPHS, name)) for name in test_names]

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

print(f"n graphs train {len(train)}")
print(f"n graphs val {len(val)}")
print(f"n graphs test {len(test)}")

train = DataLoader(train, params_GCN['batch_size'])
val = DataLoader(val, params_GCN['batch_size'])
test = DataLoader(test, params_GCN['batch_size'])

model = GCN_is_male().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params_GCN['lr'],
    weight_decay=params_GCN['weight_decay'])
loss_fun = torch.nn.BCELoss()


if __name__ == '__main__':

    mlflow.set_experiment('GCN')
    mlflow.start_run()

    list_out_test = []
    list_y_test = []
    for epoch in range(1, params_GCN['n_epochs']+1):

        list_out_train = []
        list_y_train = []

        list_out_val = []
        list_y_val = []

        print(f"[Epoch: {epoch}]")

        model.train()
        for batch in tqdm(
                train, total=len(train), colour='green'):

            out = model(batch).reshape(-1)
            y = batch.y.type(torch.cuda.FloatTensor)
            loss = loss_fun(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out = [
                1 if x.item() >= params_GCN['threshold'] else 0 for x in out]
            y = [x.item() for x in y]
            list_out_train.extend(out)
            list_y_train.extend(y)
        f1_train = f1_score(y_true=list_y_train, y_pred=list_out_train)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                    val, total=len(val), colour='green'):

                out = model(batch).reshape(-1)
                y = batch.y.type(torch.cuda.FloatTensor)

                out = [x.item() for x in out]
                out = [1 if x >= params_GCN['threshold'] else 0 for x in out]
                y = [x.item() for x in y]
                list_out_val.extend(out)
                list_y_val.extend(y)
        f1_val = f1_score(y_true=list_y_val, y_pred=list_out_val)

        mlflow.log_metric(key='f1_train', value=f1_train, step=epoch)
        mlflow.log_metric(key='f1_val', value=f1_val, step=epoch)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
                test, total=len(test), colour='green'):

            out = model(batch).reshape(-1)
            y = batch.y.type(torch.cuda.FloatTensor)

            out = [
                1 if x.item() >= params_GCN['threshold'] else 0 for x in out]
            y = [x.item() for x in y]

            list_out_test.extend(out)
            list_y_test.extend(y)
    f1_test = f1_score(y_true=list_y_test, y_pred=list_out_test)

    mlflow.log_metric(key='f1_test', value=f1_test)

    conf_matrix = confusion_matrix(list_y_test, list_out_test)
    conf_matrix_norm = confusion_matrix(
        list_y_test, list_out_test, normalize='true')
    plot = ConfusionMatrixDisplay(conf_matrix).plot().figure_
    plot_norm = ConfusionMatrixDisplay(conf_matrix_norm).plot().figure_
    plot.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix.png'))
    plot_norm.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm.png'))

    save_f(filename=os.path.join(PATH_DATA_INTERIM, 'model.pkl'), obj=model)
    mlflow.log_artifact(os.path.join(PATH_DATA_INTERIM, 'model.pkl'))

    mlflow.log_params(params=params_GCN)

    mlflow.end_run()
