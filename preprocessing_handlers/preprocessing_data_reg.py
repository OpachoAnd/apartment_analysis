import pandas as pd

# df = pd.read_excel(r'apartments_3.xlsx')


def preproc_excel(df: pd.DataFrame):
    df_copy = df.copy(deep=True)
    df_copy = df_copy.drop(columns=['Unnamed: 0', 'blanc', 'Примечание', 'Телефон', 'E-mail'])
    # Удаление мусорных строк с пустыми значениями в данных столбцах
    df_copy = df_copy.dropna(subset=['Адрес', 'о', 'цена (т.р.)'])
    # Замена значений NaN на 'Пусто'
    df_copy = df_copy.fillna('Пусто')
    # Добавление признака 'тип дома'
    df_copy[['Тип квартиры', 'тип дома']] = df_copy['Тип квартиры'].str.split(' ', expand=True, n=1)
    # Добавление признака 'всего этажей'
    df_copy[['Этаж', 'Всего этажей']] = df_copy['Этаж'].str.split('/', expand=True, n=1)
    # Приведение к нижнему регистру
    df_copy['Адрес'] = df_copy['Адрес'].str.lower()
    # Отделение номера дома от улицы
    df_copy['Адрес'] = df_copy['Адрес'].str.split('^(.+)\s(\S+)$', n=1, expand=True)[1]
    # Чистка адреса для приведение к одному виду
    df_copy['Адрес'] = df_copy['Адрес'].str.replace(r'\bул ?(. )?', '', regex=True)
    df_copy['Район'] = df_copy['Район'].str.lower()
    # df_copy = df_copy.fillna('Пусто')
    df_copy['Район'] = df_copy['Район'].fillna('Пусто')  # НАДО ЛИ ?
    # удаление фразы 'район'
    df_copy['Район'] = df_copy['Район'].str.replace(r' район', '', regex=True)
    # удаление, т.к. левый берег только 1 сэмпл
    df_copy['Район'] = df_copy['Район'].str.replace(r'ленинский [(]левый берег[)]', 'ленинский', regex=True)
    return df_copy
