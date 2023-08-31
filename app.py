from preprocessing_handlers.preprocessing_data_reg import preproc_excel
from train_handlers.train_model_apartment import TrainModel
import pandas as pd
import joblib
import numpy as np
import json

from flask import Flask, request, redirect, flash, render_template
from werkzeug.utils import secure_filename

TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMEOUT = 300
N_THREADS = 4
N_FOLDS = 5
TARGET_NAME = 'TARGET'

# train_model_reg = TrainModel(timeout=TIMEOUT,
#                              test_size=TEST_SIZE,
#                              random_state=RANDOM_STATE,
#                              n_threads=N_THREADS,
#                              n_folds=N_FOLDS)

# train_model_class = TrainModel(timeout=TIMEOUT * 7,
#                                test_size=TEST_SIZE,
#                                random_state=RANDOM_STATE,
#                                n_threads=N_THREADS,
#                                n_folds=N_FOLDS)

app = Flask(__name__)
app.secret_key = b'_5#y2L"A4Q8z\n\xec]/'


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        train_inference = request.form.get(key='train_inference')
        if train_inference == 'train':
            train_model = TrainModel(test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE,
                                     n_threads=N_THREADS,
                                     n_folds=N_FOLDS)

            df = pd.read_excel(r'apartments_3.xlsx')
            df = preproc_excel(df)
            type_task = request.form.get(key='type_task')
            if type_task == 'reg':
                train_model.train_reg(df, 'цена (т.р.)', TIMEOUT)
            elif type_task == 'multiclass':
                train_model.train_class(df, 'target', TIMEOUT * 3)

    return render_template('load_data.html', error=None)


@app.route('/regression', methods=['GET', 'POST'])
def reg_page():
    if request.method == 'POST':
        post_json = request.get_json()
        test = pd.DataFrame.from_dict(post_json)
        model = joblib.load('model_reg.pkl')
        return str(model.predict(test))
    return 'not data'


@app.route('/classification', methods=['GET', 'POST'])
def class_page():
    if request.method == 'POST':
        post_json = request.get_json()
        test = pd.DataFrame.from_dict(post_json)
        model = joblib.load('model_class_2.pkl')
        # print(test)
        # return (test)
        # return str(model.predict(test))
        mapping = model.reader.class_mapping

        def map_class(x):
            return mapping[x]

        mapped = np.vectorize(map_class)
        # mapped(train_data['target'].values)
        pred = model.predict(test)
        pred_list = pred[0].data
        # print(type(pred_list))
        # max_value = max(pred_list)
        max_index = pred_list.argmax()  # pred_list.index(max_value)
        # print(mapping[max_index])
        print(max_index)
        print(pred_list)
        print(mapping)

        print(list(mapping.keys())[list(mapping.values()).index(max_index)])
        # pred_np = np.ndarray(pred[0])
        # print(pred)
    return 'not data'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

    # df = pd.read_excel(r'apartments_3.xlsx')
    # df = preproc_excel(df)

    # print(df)

    # train_model_reg.train_reg(df, 'цена (т.р.)')
    # train_model_class.train_class(df, 'target')
    # m = joblib.load('model_class_3.pkl')
    # mapping = m.reader.class_mapping
    #
    # print(mapping)
    # def map_class(x):
    #     return mapping[x]


    # mapped = np.vectorize(map_class)
    # mapped(train_data['target'].values)

# {
#     "Тип квартиры": ["трехкомнатная"],
#     "Район": ["пусто"],
#     "Адрес": ["грязнова"],
#     "Этаж": ["5"],
#     "о": ["56.0"],
#     "ж": ["40.0"],
#     "к": ["6.0"],
#     "Агентство": ["РиоЛюкс"],
#     "тип дома": [""],
#     "Всего этажей": ["5"]
# }