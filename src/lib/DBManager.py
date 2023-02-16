import pandas as pd


class DBManager:

    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()

    @staticmethod
    def __get_dtypes_for_db(df: pd.DataFrame) -> str:
        colType_list = []
        dtypes = df.dtypes.to_dict()
        dtypes = {col: str(val) for col, val in dtypes.items()}
        for colname, coltype in dtypes.items():
            if coltype == 'datetime64[ns]':
                coltype = 'date'
            elif 'int' in coltype:
                coltype = 'INTEGER'
            elif 'float' in coltype:
                coltype = 'REAL'
            elif 'object' in coltype:
                coltype = 'TEXT'
            else:
                pass
            colType_list.append(colname + ' ' + coltype)
        dtypes = ', '.join(colType_list)
        dtypes = '(' + dtypes + ')'
        return dtypes

    @staticmethod
    def __prepare_before_db(df: pd.DataFrame, if_exist: str):
        if if_exist == 'replace':
            df = df.fillna('NULL')
            cols_datetime = df.select_dtypes(include='datetime64[ns]').columns
            df[cols_datetime] = df[cols_datetime]\
                .apply(lambda x: x.astype(str))
            df = list(df.itertuples(index=False, name=None))
            df = [str(x) for x in df]
            df = ', '.join(df)
        if if_exist == 'append':
            df = df.fillna('NULL')
            cols_datetime = df.select_dtypes(include='datetime64[ns]').columns
            df[cols_datetime] = df[cols_datetime]\
                .apply(lambda x: x.astype(str))
            df = list(df.itertuples(index=False, name=None))
        return df

    @staticmethod
    def __table_exist(name, connector, cursor) -> bool:
        cursor.execute(
            f"""
            SELECT
                COUNT(name)
            FROM
                sqlite_master
            WHERE
                type = 'table' and name = '{name}'
            """)
        if cursor.fetchone()[0] == 1:
            response = True
        else:
            response = False
        connector.commit()
        return response

    def write_df(
            self, df: pd.DataFrame, table_name: str, if_exist: str) -> None:
        """
        if_exist types:
        1) replace;
        2) append.
        """

        colnames_coltypes_for_db = self.__get_dtypes_for_db(df)

        if if_exist == 'replace':
            data = self.__prepare_before_db(df, 'replace')
            self.cur.execute(
                f"""
                DROP TABLE IF EXISTS {table_name};
                """
            )
            self.cur.execute(
                f"""
                CREATE TABLE
                    {table_name}{colnames_coltypes_for_db}
                """
            )
            self.cur.execute(
                f"INSERT INTO {table_name} VALUES({data})")
            self.conn.commit()

        if if_exist == 'append':
            data = self.__prepare_before_db(df, 'append')
            vals = ', '.join(['?'] * len(data[0]))
            if self.__table_exist(table_name, self.conn, self.cur):
                self.cur.executemany(
                    f"""
                    INSERT INTO {table_name} VALUES ({vals})
                    """, data)
                self.conn.commit()
            else:
                self.cur.execute(
                    f"""
                    CREATE TABLE
                        {table_name}{colnames_coltypes_for_db}
                    """
                )
                self.cur.executemany(
                    f"""
                    INSERT INTO {table_name} VALUES ({vals})
                    """, data)
                self.conn.commit()

    def read_data_as_df(self, query: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"{query}",
            con=self.conn
        )
        return df
