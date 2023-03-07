# %%
import mlflow
from lib.utils import load_f
from base.settings import PATH_DATA_INTERIM
# %%
with mlflow.start_run():
    model_age = mlflow.artifacts.download_artifacts(
        artifact_uri='runs:/3c70a74a290e446890f749b99548acc5/model.pkl')
    print(model_age)
# %%
