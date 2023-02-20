import os
import torch
import mlflow
from tqdm import tqdm
from lib.utils import save_f
from lib.utils import load_f
from base.settings import PATH_DATA_INTERIM
from lib.models import GCN, params_GCN
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

print(f"n graphs train {len(train)}")
print(f"n graphs val {len(val)}")
print(f"n graphs test {len(test)}")

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
        list_loss_test = []
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

    y_hat = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
                test, total=len(test), colour='green'):

            out = model(batch)
            y = batch.y.type(torch.cuda.ByteTensor)
            loss = loss_fun(out, y)

            list_loss_test.append(loss.item())

            hat = list(torch.argmax(out, dim=1).cpu().numpy())
            true = list(y.cpu().numpy())
            y_hat.extend(hat)
            y_true.extend(true)

    loss_test = sum(list_loss_test) / len(list_loss_test)
    mlflow.log_metric(key='loss test', value=loss_test)

    labels = ['19-25', '26-35', '36-45', '46-55', '56-65', '66-inf']
    conf_matrix = confusion_matrix(y_true, y_hat)
    conf_matrix_norm = confusion_matrix(y_true, y_hat, normalize='true')
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

    save_f(filename=os.path.join(PATH_DATA_INTERIM, 'model.pkl'), obj=model)
    mlflow.log_artifact(os.path.join(PATH_DATA_INTERIM, 'model.pkl'))

    mlflow.log_params(params=params_GCN)

    mlflow.end_run()
