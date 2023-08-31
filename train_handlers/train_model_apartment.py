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
        # train_data, test_data = train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')

        # task = Task('reg', loss='mae', metric='mae')
        # roles = {'target': target_column,  # 'цена (т.р.)',
        #          'category': ['Тип квартиры', 'Район', 'Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей'],
        #          'numeric': ['о', 'ж', 'к']}
        #
        # automl = TabularAutoML(task=task,
        #                        timeout=self.timeout,
        #                        cpu_limit=self.n_threads,
        #                        reader_params={'n_jobs': self.n_threads,
        #                                       'cv': self.n_folds,
        #                                       'random_state': self.random_state})
        # automl.fit_predict(train_data, roles=roles, verbose=1)
        # test_pred = automl.predict(test_data)
        automl, out_of_fold_pred, test_pred, roles = self.model(task_name='reg',
                                                                train_data=train_data,
                                                                test_data=test_data,
                                                                timeout=timeout,
                                                                target_column=target_column)

        print(f'HOLDOUT score: {mean_absolute_error(test_data[roles["target"]].values, test_pred.data[:, 0])}')
        joblib.dump(automl, 'model_reg.pkl')
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

        # task = Task('multiclass')
        # roles = {'target': target_column,
        #          'category': ['Тип квартиры', 'Район', 'Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей'],
        #          'numeric': ['о', 'ж', 'к']}
        #
        # automl = TabularAutoML(task=task,
        #                        timeout=self.timeout * 3,
        #                        cpu_limit=self.n_threads,
        #                        reader_params={'n_jobs': self.n_threads,
        #                                       'cv': self.n_folds,
        #                                       'random_state': self.random_state})
        #
        # out_of_fold_pred = automl.fit_predict(train_data, roles=roles, verbose=1)
        # test_pred = automl.predict(test_data)
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
        joblib.dump(automl, 'model_class_3.pkl')
        return automl


# def train_reg(df: pd.DataFrame):
#     train_data, test_data = train_test_split(df,
#                                              test_size=TEST_SIZE,
#                                              random_state=RANDOM_STATE)
#     # train_data, test_data = train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
#     print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')
#
#     task = Task('reg', loss='mae', metric='mae')
#
#     roles = {'target': 'цена (т.р.)',
#              'category': ['Тип квартиры', 'Район', 'Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей'],
#              'numeric': ['о', 'ж', 'к']}
#
#     automl = TabularAutoML(task=task,
#                            timeout=TIMEOUT,
#                            cpu_limit=N_THREADS,
#                            reader_params={'n_jobs': N_THREADS,
#                                           'cv': N_FOLDS,
#                                           'random_state': RANDOM_STATE})
#     automl.fit_predict(train_data, roles=roles, verbose=1)
#     test_pred = automl.predict(test_data)
#
#     print(f'HOLDOUT score: {mean_absolute_error(test_data[roles["target"]].values, test_pred.data[:, 0])}')
#     joblib.dump(automl, 'model.pkl')
#     return automl
#
#
# def train_class(df: pd.DataFrame):
#     train_data, test_data = train_test_split(df,
#                                              test_size=TEST_SIZE,
#                                              random_state=RANDOM_STATE,
#                                              shuffle=True)
#     print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')
#
#     task = Task('multiclass')
#     roles = {'target': 'target',
#              # 'category': ['Тип квартиры', 'Район', 'Адрес', 'Агентство', 'тип дома', 'Этаж', 'Всего этажей'],
#              'category': ['Тип квартиры'],
#              'numeric': ['о', 'ж', 'к', '']}
#
#     automl = TabularAutoML(task=task,
#                            timeout=TIMEOUT * 3,
#                            cpu_limit=N_THREADS,
#                            reader_params={'n_jobs': N_THREADS,
#                                           'cv': N_FOLDS,
#                                           'random_state': RANDOM_STATE})
#
#     out_of_fold_pred = automl.fit_predict(train_data, roles=roles, verbose=1)
#     test_pred = automl.predict(test_data)
#
#     mapping = automl.reader.class_mapping
#
#     def map_class(x):
#         return mapping[x]
#
#     mapped = np.vectorize(map_class)
#     mapped(train_data['target'].values)
#
#     print(f'HOLDOUT score: {log_loss(mapped(test_data[roles["target"]].values), test_pred.data)}')
#     joblib.dump(automl, 'model_class.pkl')
#     return automl
