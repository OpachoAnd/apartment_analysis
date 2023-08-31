import warnings

import joblib
# Essential DS libraries
import numpy as np
import pandas as pd
# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

from db.requests_db import DBCommands

TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMEOUT = 300
N_THREADS = 4
N_FOLDS = 5
TARGET_NAME = 'TARGET'

warnings.simplefilter(action='ignore', category=FutureWarning)


class TrainModel():
    def __init__(self, test_size=0.2, random_state=42, n_threads=4, n_folds=5):
        self.test_size = test_size
        self.random_state = random_state
        # self.timeout = timeout
        self.n_threads = n_threads
        self.n_folds = n_folds

    def model(self, task_name, target_column, train_data, test_data, timeout):
        task = Task(task_name)
        if task_name == 'reg':
            roles = {'target': target_column,
                     'category': ['Тип квартиры', 'Район', 'Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей'],
                     'numeric': ['о', 'ж', 'к']}
        elif task_name == 'multiclass':
            roles = {'target': target_column,
                     'category': ['Тип квартиры'],
                     'numeric': ['о', 'ж', 'к', 'цена (т.р.)'],
                     'drop': ['Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей']}

        automl = TabularAutoML(task=task,
                               timeout=timeout,  # self.timeout * 3,
                               cpu_limit=self.n_threads,
                               reader_params={'n_jobs': self.n_threads,
                                              'cv': self.n_folds,
                                              'random_state': self.random_state})

        out_of_fold_pred = automl.fit_predict(train_data, roles=roles, verbose=1)
        print(test_data.columns)
        test_pred = automl.predict(test_data)
        return automl, out_of_fold_pred, test_pred, roles

    def train_reg(self, df, target_column, timeout):
        df_copy = df.copy(deep=True)
        train_data, test_data = train_test_split(df_copy,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state)
        print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')

        automl, out_of_fold_pred, test_pred, roles = self.model(task_name='reg',
                                                                train_data=train_data,
                                                                test_data=test_data,
                                                                timeout=timeout,
                                                                target_column=target_column)

        print(f'HOLDOUT score: {mean_absolute_error(test_data[roles["target"]].values, test_pred.data[:, 0])}')

        model_dump = pickle.dumps(automl)
        db_commands = DBCommands()
        # db_commands.update_model_to_db(model_dump)
        db_commands.update_model_reg_to_db(model_dump)

        joblib.dump(automl, 'model_reg_new.pkl')
        return automl

    def train_class(self, df: pd.DataFrame, target_column: str, timeout):
        df_copy = df.copy(deep=True)
        # Так как ЛАМА не принимает другое название столбца
        df_copy.rename(columns={'Район': 'target'}, inplace=True)
        train_data, test_data = train_test_split(df_copy,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state,
                                                 shuffle=True)
        print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')

        automl, out_of_fold_pred, test_pred, roles = self.model(task_name='multiclass',
                                                                train_data=train_data,
                                                                test_data=test_data,
                                                                timeout=timeout,
                                                                target_column=target_column)

        mapping = automl.reader.class_mapping

        def map_class(x):
            return mapping[x]

        mapped = np.vectorize(map_class)
        mapped(train_data['target'].values)

        print(f'HOLDOUT score: {log_loss(mapped(test_data[roles["target"]].values), test_pred.data)}')

        model_dump = pickle.dumps(automl)
        db_commands = DBCommands()
        # db_commands.update_model_to_db(model_dump)
        db_commands.update_model_class_to_db(model_dump)

        joblib.dump(automl, 'model_class_new.pkl')
        return automl
