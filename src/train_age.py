import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import save_f, load_f
from base.settings import PATH_DATA_INTERIM
from lib.params_age import params, params_model
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn.aggr import MaxAggregation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

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
loss_fun = torch.nn.CrossEntropyLoss()


if __name__ == '__main__':

    mlflow.set_experiment('GNN')
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
            out = torch.softmax(out, dim=0)
            y = batch.y.type(torch.cuda.ByteTensor)
            loss = loss_fun(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out = list(torch.argmax(out, dim=1).cpu().numpy())
            y = list(y.cpu().numpy())

            list_out_train.extend(out)
            list_y_train.extend(y)

        f1_train_micro = f1_score(
            y_true=list_y_train,
            y_pred=list_out_train,
            average='micro')
        f1_train_macro = f1_score(
            y_true=list_y_train,
            y_pred=list_out_train,
            average='macro')

        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                    val, total=len(val), colour='green'):

                out = model(batch.x, batch.edge_index)
                out = agg_fun(out, batch.batch)
                out = torch.softmax(out, dim=0)
                y = batch.y.type(torch.cuda.ByteTensor)

                out = list(torch.argmax(out, dim=1).cpu().numpy())
                y = list(y.cpu().numpy())

                list_out_val.extend(out)
                list_y_val.extend(y)

        f1_val_micro = f1_score(
            y_true=list_y_val,
            y_pred=list_out_val,
            average='micro')
        f1_val_macro = f1_score(
            y_true=list_y_val,
            y_pred=list_out_val,
            average='macro')

        mlflow.log_metric(
            key='f1_train_micro',
            value=f1_train_micro,
            step=epoch)
        mlflow.log_metric(
            key='f1_train_macro',
            value=f1_train_macro,
            step=epoch)
        mlflow.log_metric(
            key='f1_val_micro',
            value=f1_val_micro,
            step=epoch)
        mlflow.log_metric(
            key='f1_val_macro',
            value=f1_val_macro,
            step=epoch)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
                test, total=len(test), colour='green'):

            out = model(batch.x, batch.edge_index)
            out = model(batch.x, batch.edge_index)
            out = agg_fun(out, batch.batch)
            out = torch.softmax(out, dim=0)
            y = batch.y.type(torch.cuda.ByteTensor)

            out = list(torch.argmax(out, dim=1).cpu().numpy())
            y = list(y.cpu().numpy())
            list_out_test.extend(out)
            list_y_test.extend(y)

    f1_test_micro = f1_score(
        y_true=list_y_test,
        y_pred=list_out_test,
        average='micro')
    f1_test_macro = f1_score(
        y_true=list_y_test,
        y_pred=list_out_test,
        average='macro')

    mlflow.log_metric(
        key='f1_test_micro',
        value=f1_test_micro,
        step=epoch)
    mlflow.log_metric(
        key='f1_test_macro',
        value=f1_test_macro,
        step=epoch)

    labels = ['19-25', '26-35', '36-45', '46-55', '56-65', '66-inf']
    conf_matrix = confusion_matrix(list_y_test, list_out_test)
    conf_matrix_norm = confusion_matrix(
        list_y_test, list_out_test, normalize='true')
    plot = ConfusionMatrixDisplay(
        conf_matrix, display_labels=labels).plot().figure_
    plot_norm = ConfusionMatrixDisplay(
        conf_matrix_norm, display_labels=labels).plot().figure_
    plot.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix.png'))
    plot_norm.savefig(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix.png'))
    mlflow.log_artifact(
        os.path.join(PATH_DATA_INTERIM, 'conf_matrix_norm.png'))

    save_f(
        filename=os.path.join(PATH_DATA_INTERIM, 'model.pkl'),
        obj=model)

    mlflow.log_artifact(os.path.join(PATH_DATA_INTERIM, 'model.pkl'))

    mlflow.log_params(params=params)

    mlflow.end_run()
