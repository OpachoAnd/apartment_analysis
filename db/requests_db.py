import json

import pandas as pd
from sqlalchemy.orm import sessionmaker

from db.config import ENGINE
from db.models import Base, ML_table
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

    def update_model_reg_to_db(self, model):
        ml_table = self.get(ML_table, ml_table_id=1)
        ml_table.model_lama_reg = model
        self.pool.commit()

    def update_model_class_to_db(self, model):
        ml_table = self.get(ML_table, ml_table_id=1)
        ml_table.model_lama_class = model
        self.pool.commit()

    def get(self, model, **kwargs):
        instance = self.pool.query(model).filter_by(**kwargs).first()
        return instance if instance else None

    # def get_df(self, ml_table_id):
    #     pass

    def get_model(self, model_id):
        return self.get(ML_table, ml_table_id=model_id)
