import pickle

from preprocessing_handlers.preprocessing_data_reg import preproc_excel
from train_handlers.train_model_apartment import TrainModel
import pandas as pd
import numpy as np
import json
from db.requests_db import DBCommands

from flask import Flask, request, redirect, flash, render_template

TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMEOUT = 300
N_THREADS = 4
N_FOLDS = 5
TARGET_NAME = 'TARGET'


app = Flask(__name__)
app.secret_key = b'_5#y2L"A4Q8z\n\xec]/'

db_commands = DBCommands()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        train_inference = request.form.get(key='train_inference')
        if train_inference == 'train':
            train_model = TrainModel(test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE,
                                     n_threads=N_THREADS,
                                     n_folds=N_FOLDS)
            df_from_bd = db_commands.get_model(model_id=1).data_df

            df_bytes = json.loads(df_from_bd)
            df = pd.DataFrame.from_dict(df_bytes)
            print(df)
            # df = pd.read_excel(r'apartments_3.xlsx')
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

        model_byte = db_commands.get_model(model_id=1).model_lama_reg
        model = pickle.loads(model_byte)
        # model = joblib.load('model_reg.pkl')
        return {'Цена квартиры': str(model.predict(test)[0].data[0])}
    return 'not data'


@app.route('/classification', methods=['GET', 'POST'])
def class_page():
    if request.method == 'POST':
        post_json = request.get_json()
        test = pd.DataFrame.from_dict(post_json)
        model_byte = db_commands.get_model(model_id=1).model_lama_class
        model = pickle.loads(model_byte)
        # model = joblib.load('model_class_2.pkl')

        mapping = model.reader.class_mapping

        def map_class(x):
            return mapping[x]

        mapped = np.vectorize(map_class)

        pred = model.predict(test)
        pred_list = pred[0].data

        max_index = pred_list.argmax()


        return {'Район': list(mapping.keys())[list(mapping.values()).index(max_index)]}

    return 'not data'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
