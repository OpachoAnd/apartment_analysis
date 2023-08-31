import json

import pandas as pd
from sqlalchemy.orm import sessionmaker

from db.config import ENGINE
from db.models import Base, ML_table
import io
import sqlalchemy


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DBManager(metaclass=SingletonMeta):
    connection = None

    @classmethod
    def connect(cls):
        if cls.connection is None:
            cls.connection = sessionmaker(bind=ENGINE)()
        return cls.connection


class DBCommands:
    def __init__(self):
        self.pool = DBManager.connect()

    @staticmethod
    def create_tables():
        """
        Создает таблицы в БД
        """
        Base.metadata.create_all(ENGINE)

    @staticmethod
    def drop_tables():
        Base.metadata.drop_all(ENGINE)

    def add_df_to_db(self,
                     df):
        ml_table = ML_table(data_df=df)
        self.pool.add(ml_table)
        self.pool.commit()

    def get(self, model, **kwargs):
        instance = self.pool.query(model).filter_by(**kwargs).first()
        return instance if instance else None

    def get_model(self, model_id):
        return self.get(ML_table, ml_table_id=model_id)


if __name__ == '__main__':
    # Создание текущих таблиц при начале работы с базой данных
    # DBCommands.create_tables()
    db_commands = DBCommands()

    towrite = io.BytesIO()
    df = pd.read_excel('/home/opacho/Документы/GitHub/apartment_analysis/apartments_3.xlsx')
    # df.to_excel(towrite, index=False)
    # towrite.seek(0)
    # bytes_data = towrite.getvalue()

    df_byte = df.to_json().encode()
    data = json.loads(df_byte)
    q = pd.DataFrame.from_dict(data)
    print(q)
    # db_commands.add_df_to_db(df_byte)


    # db_commands.add_text_to_db(text='hello ML')
    # q = db_commands.get_model(model_id=1)
    # print(q.model.tobytes())

    # table_names = sqlalchemy.inspect(ML_table)
    # attr_names = [c_attr.key for c_attr in table_names.mapper.column_attrs]
    # print(attr_names)
