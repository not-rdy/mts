# %%
from lib.DBManager import DBManager
from base.settings import connector

db = DBManager(connector)
# %%
data = db.read_data_as_df(
    """
    select
        *
    from
        test_agg
    """
)
# %%
data.isna().sum()
# %%
