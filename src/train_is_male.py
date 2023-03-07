import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import save_f, load_f
from base.settings import PATH_DATA_INTERIM
from lib.params_is_male import params, params_model, params_agg_lstm
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn.aggr import LSTMAggregation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

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
agg_fun = LSTMAggregation(**params_agg_lstm).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params['lr'],
    weight_decay=params['weight_decay'])
loss_fun = torch.nn.CrossEntropyLoss()


if __name__ == '__main__':

    mlflow.set_experiment('GNN')
    mlflow.start_run()

    list_out_03_test = []
    list_out_04_test = []
    list_out_05_test = []
    list_out_06_test = []
    list_out_07_test = []
    list_y_test = []
    for epoch in range(1, params['n_epochs']+1):

        list_out_03_train = []
        list_out_04_train = []
        list_out_05_train = []
        list_out_06_train = []
        list_out_07_train = []
        list_y_train = []

        list_out_03_val = []
        list_out_04_val = []
        list_out_05_val = []
        list_out_06_val = []
        list_out_07_val = []
        list_y_val = []

        print(f"[Epoch: {epoch}]")

        model.train()
        agg_fun.train()
        for batch in tqdm(
                train, total=len(train), colour='green'):

            out = model(batch.x, batch.edge_index)
            out = agg_fun(out, batch.batch)
            out = torch.sigmoid(out)
            y = batch.y.type(torch.cuda.ByteTensor)
            loss = loss_fun(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_03 = [1 if x.item() >= 0.3 else 0 for x in out]
            out_04 = [1 if x.item() >= 0.4 else 0 for x in out]
            out_05 = [1 if x.item() >= 0.5 else 0 for x in out]
            out_06 = [1 if x.item() >= 0.6 else 0 for x in out]
            out_07 = [1 if x.item() >= 0.7 else 0 for x in out]
            y = list(y.cpu().numpy())

            list_out_03_train.extend(out_03)
            list_out_04_train.extend(out_04)
            list_out_05_train.extend(out_05)
            list_out_06_train.extend(out_06)
            list_out_07_train.extend(out_07)
            list_y_train.extend(y)

        f1_train_03 = f1_score(
            y_true=list_y_train,
            y_pred=list_out_03_train)
        f1_train_04 = f1_score(
            y_true=list_y_train,
            y_pred=list_out_04_train)
        f1_train_05 = f1_score(
            y_true=list_y_train,
            y_pred=list_out_05_train)
        f1_train_06 = f1_score(
            y_true=list_y_train,
            y_pred=list_out_06_train)
        f1_train_07 = f1_score(
            y_true=list_y_train,
            y_pred=list_out_07_train)

        model.eval()
        agg_fun.eval()
        with torch.no_grad():
            for batch in tqdm(
                    val, total=len(val), colour='green'):

                out = model(batch.x, batch.edge_index)
                out = agg_fun(out, batch.batch)
                out = torch.sigmoid(out)
                y = batch.y.type(torch.cuda.ByteTensor)

                out_03 = [1 if x.item() >= 0.3 else 0 for x in out]
                out_04 = [1 if x.item() >= 0.4 else 0 for x in out]
                out_05 = [1 if x.item() >= 0.5 else 0 for x in out]
                out_06 = [1 if x.item() >= 0.6 else 0 for x in out]
                out_07 = [1 if x.item() >= 0.7 else 0 for x in out]
                y = list(y.cpu().numpy())

                list_out_03_val.extend(out_03)
                list_out_04_val.extend(out_04)
                list_out_05_val.extend(out_05)
                list_out_06_val.extend(out_06)
                list_out_07_val.extend(out_07)
                list_y_val.extend(y)

        f1_val_03 = f1_score(
            y_true=list_y_val,
            y_pred=list_out_03_val)
        f1_val_04 = f1_score(
            y_true=list_y_val,
            y_pred=list_out_04_val)
        f1_val_05 = f1_score(
            y_true=list_y_val,
            y_pred=list_out_05_val)
        f1_val_06 = f1_score(
            y_true=list_y_val,
            y_pred=list_out_06_val)
        f1_val_07 = f1_score(
            y_true=list_y_val,
            y_pred=list_out_07_val)

        mlflow.log_metric(
            key='f1_train_03', value=f1_train_03, step=epoch)
        mlflow.log_metric(
            key='f1_train_04', value=f1_train_04, step=epoch)
        mlflow.log_metric(
            key='f1_train_05', value=f1_train_05, step=epoch)
        mlflow.log_metric(
            key='f1_train_06', value=f1_train_06, step=epoch)
        mlflow.log_metric(
            key='f1_train_07', value=f1_train_07, step=epoch)
        mlflow.log_metric(
            key='f1_val_03', value=f1_val_03, step=epoch)
        mlflow.log_metric(
            key='f1_val_04', value=f1_val_04, step=epoch)
        mlflow.log_metric(
            key='f1_val_05', value=f1_val_05, step=epoch)
        mlflow.log_metric(
            key='f1_val_06', value=f1_val_06, step=epoch)
        mlflow.log_metric(
            key='f1_val_07', value=f1_val_07, step=epoch)

    model.eval()
    agg_fun.eval()
    with torch.no_grad():
        for batch in tqdm(
                test, total=len(test), colour='green'):

            out = model(batch.x, batch.edge_index)
            out = model(batch.x, batch.edge_index)
            out = agg_fun(out, batch.batch)
            out = torch.sigmoid(out)
            y = batch.y.type(torch.cuda.ByteTensor)

            out_03 = [1 if x.item() >= 0.3 else 0 for x in out]
            out_04 = [1 if x.item() >= 0.4 else 0 for x in out]
            out_05 = [1 if x.item() >= 0.5 else 0 for x in out]
            out_06 = [1 if x.item() >= 0.6 else 0 for x in out]
            out_07 = [1 if x.item() >= 0.7 else 0 for x in out]
            y = list(y.cpu().numpy())

            list_out_03_test.extend(out_03)
            list_out_04_test.extend(out_04)
            list_out_05_test.extend(out_05)
            list_out_06_test.extend(out_06)
            list_out_07_test.extend(out_07)
            list_y_test.extend(y)

    f1_test_03 = f1_score(
        y_true=list_y_test,
        y_pred=list_out_03_test)
    f1_test_04 = f1_score(
        y_true=list_y_test,
        y_pred=list_out_04_test)
    f1_test_05 = f1_score(
        y_true=list_y_test,
        y_pred=list_out_05_test)
    f1_test_06 = f1_score(
        y_true=list_y_test,
        y_pred=list_out_06_test)
    f1_test_07 = f1_score(
        y_true=list_y_test,
        y_pred=list_out_07_test)

    mlflow.log_metric(
        key='f1_test_03', value=f1_test_03, step=epoch)
    mlflow.log_metric(
        key='f1_test_04', value=f1_test_04, step=epoch)
    mlflow.log_metric(
        key='f1_test_05', value=f1_test_05, step=epoch)
    mlflow.log_metric(
        key='f1_test_06', value=f1_test_06, step=epoch)
    mlflow.log_metric(
        key='f1_test_07', value=f1_test_07, step=epoch)

    conf_matrix_03 = confusion_matrix(list_y_test, list_out_03_test)
    conf_matrix_04 = confusion_matrix(list_y_test, list_out_04_test)
    conf_matrix_05 = confusion_matrix(list_y_test, list_out_05_test)
    conf_matrix_06 = confusion_matrix(list_y_test, list_out_06_test)
    conf_matrix_07 = confusion_matrix(list_y_test, list_out_07_test)

    conf_matrix_03_norm = confusion_matrix(
        list_y_test, list_out_03_test, normalize='true')
    conf_matrix_04_norm = confusion_matrix(
        list_y_test, list_out_04_test, normalize='true')
    conf_matrix_05_norm = confusion_matrix(
        list_y_test, list_out_05_test, normalize='true')
    conf_matrix_06_norm = confusion_matrix(
        list_y_test, list_out_06_test, normalize='true')
    conf_matrix_07_norm = confusion_matrix(
        list_y_test, list_out_07_test, normalize='true')

    plot_03 = ConfusionMatrixDisplay(conf_matrix_03).plot().figure_
    plot_04 = ConfusionMatrixDisplay(conf_matrix_04).plot().figure_
    plot_05 = ConfusionMatrixDisplay(conf_matrix_05).plot().figure_
    plot_06 = ConfusionMatrixDisplay(conf_matrix_06).plot().figure_
    plot_07 = ConfusionMatrixDisplay(conf_matrix_07).plot().figure_
    plot_norm_03 = ConfusionMatrixDisplay(conf_matrix_03_norm).plot().figure_
    plot_norm_04 = ConfusionMatrixDisplay(conf_matrix_04_norm).plot().figure_
    plot_norm_05 = ConfusionMatrixDisplay(conf_matrix_05_norm).plot().figure_
    plot_norm_06 = ConfusionMatrixDisplay(conf_matrix_06_norm).plot().figure_
    plot_norm_07 = ConfusionMatrixDisplay(conf_matrix_07_norm).plot().figure_

    plot_03.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_03.png'))
    plot_04.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_04.png'))
    plot_05.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_05.png'))
    plot_06.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_06.png'))
    plot_07.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_07.png'))
    plot_norm_03.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_03.png'))
    plot_norm_04.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_04.png'))
    plot_norm_05.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_05.png'))
    plot_norm_06.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_06.png'))
    plot_norm_07.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_07.png'))

    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_03.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_04.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_05.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_06.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_07.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_03.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_04.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_05.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_06.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm_07.png'))

    save_f(filename=os.path.join(PATH_DATA_INTERIM, 'model.pkl'), obj=model)
    mlflow.log_artifact(os.path.join(PATH_DATA_INTERIM, 'model.pkl'))

    mlflow.log_params(params=params)
    mlflow.log_params(params=params_model)
    mlflow.log_params(params=params_agg_lstm)

    mlflow.end_run()
