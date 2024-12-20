import os
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import random

with open('catboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

def get_user_input():
    session_id = input("Введите session_id: ")
    if session_id == '':
        session_id = np.nan

    sites = []
    times = []

    for i in range(1, 11):  # от site1 до site10
        site = input(f"Введите site{i}: ")
        if site == '':
            site = np.nan
        time = input(f"Введите time{i}: ")
        if time == '':
            time = np.nan
        sites.append(site)
        times.append(time)

    target = input("Введите target (или оставьте пустым, если не применимо): ")
    if target == '':
        target = np.nan

    # Создаем словарь для данных
    data = {
        'session_id': session_id,
        **{f'site{i}': sites[i-1] for i in range(1, 11)},
        **{f'time{i}': times[i-1] for i in range(1, 11)},
        'target': target
    }

    return data

def get_data_Alice():

    # Загрузка данных из файла
    df = pd.read_csv("Alice_log.csv")
    
    # Выбор 10 случайных строк
    random_rows = df.sample(n=10, random_state=np.random.randint(0, 9999))
    
    # Создание DataFrame с требуемыми столбцами
    data = {
        'session_id':int( np.random.randint(0, 9999)),
        'site1': random_rows['site'].iloc[0:1].tolist(),
        'site2': random_rows['site'].iloc[1:2].tolist(),
        'site3': random_rows['site'].iloc[2:3].tolist(),
        'site4': random_rows['site'].iloc[3:4].tolist(),
        'site5': random_rows['site'].iloc[4:5].tolist(),
        'site6': random_rows['site'].iloc[5:6].tolist(),
        'site7': random_rows['site'].iloc[6:7].tolist(),
        'site8': random_rows['site'].iloc[7:8].tolist(),
        'site9': random_rows['site'].iloc[8:9].tolist(),
        'site10': random_rows['site'].iloc[9:10].tolist() ,
        'time1': random_rows['timestamp'].iloc[0:1].tolist(),
        'time2': random_rows['timestamp'].iloc[1:2].tolist(),
        'time3': random_rows['timestamp'].iloc[2:3].tolist(),
        'time4': random_rows['timestamp'].iloc[3:4].tolist(),
        'time5': random_rows['timestamp'].iloc[4:5].tolist(),
        'time6': random_rows['timestamp'].iloc[5:6].tolist(),
        'time7': random_rows['timestamp'].iloc[6:7].tolist(),
        'time8': random_rows['timestamp'].iloc[7:8].tolist(),
        'time9': random_rows['timestamp'].iloc[8:9].tolist(),
        'time10': random_rows['timestamp'].iloc[9:10].tolist() ,
        'target': 1
    }
    
    # Преобразование в DataFrame
    result_df = pd.DataFrame(data)
    
    return result_df

def get_random_user():
    # Определяем путь к папке
    folder_path = "other_user_logs"
    
    # Получаем список файлов в папке
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Выбираем случайный файл
    selected_file = random.choice(files)
    file_path = os.path.join(folder_path, selected_file)

    # Читаем файл в DataFrame (предполагается, что это CSV-файл)
    df = pd.read_csv(file_path)
    
    random_session = df.sample(n=10)  # 10 строк для сессии
    
    # Создаем DataFrame с нужными колонками
    columns = [
        'session_id', 'site1', 'time1', 'site2', 'time2', 'site3', 'time3', 
        'site4', 'time4', 'site5', 'time5', 'site6', 'time6', 'site7', 'time7', 
        'site8', 'time8', 'site9', 'time9', 'site10', 'time10', 'target'
    ]
    
    # Заполняем DataFrame данными из случайной сессии
    data = {
        'session_id': [selected_file],  # Используем название файла как session_id
        'target': 0  # Изначально target ноль
    }
    for i, row in enumerate(random_session.itertuples(), 1):
        data[f'site{i}'] = [row.site]
        data[f'time{i}'] = [row.timestamp]
    
    # Создаем DataFrame
    df_random_user = pd.DataFrame(data, columns=columns)

    return df_random_user

def save_to_dataframe(data):
    # Создаем DataFrame с полученными данными
    df = pd.DataFrame([data])
    return df

def making_more_columns(df):

    # Преобразование столбцов времени в формат datetime
    time_columns = [ 'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10']

    # Преобразование всех временных столбцов
    for col in time_columns:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
        df[f'{col}_second'] = df[col].dt.second

    # Определение самого часто посещаемого сайта
    site_columns = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 'site7', 'site8', 'site9', 'site10']
    df['most_visited_site'] = df[site_columns].apply(lambda x: x.value_counts().idxmax() if x.value_counts().max() > 1 else np.nan, axis=1)

    # Определение времени суток (утро, день, вечер)
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 1
        elif 12 <= hour < 18:
            return 2
        else:
            return 3

    df['time1_part_of_day'] = df['time1'].dt.hour.apply(time_of_day)


    for col in time_columns:
        
        # Удаляем исходные столбцы времени
        df.drop(col, axis=1, inplace=True)

    return df

def predict_it (df):
    df['session_id'] = df['session_id'].str.extract('(\d+)')

    X_new = df.drop(columns=['target'])

    predicted_target = model.predict(X_new)

    true_target = df['target'].values[0]

    if int(predicted_target[0]) == int(true_target):
        return 1 
        #return(f"Предсказание верно! Предсказанный target: {predicted_target[0]}, Истинный target: {true_target}")
    else:
        return 0
        #return(f"Предсказание неверно. Предсказанный target: {predicted_target[0]}, Истинный target: {true_target}")
  
def replace_sites_with_codes(df, site_dict_path):
    # Загрузка словаря с кодировкой сайтов
    with open(site_dict_path, 'rb') as file:
        site_dict = pickle.load(file)

    # Замена сайтов на их коды
    for col in df.columns:
        if 'site' in col:
            df[col] = df[col].map(site_dict).fillna(df[col])  # Заменяем сайт на код, если он есть в словаре

    return df

def reorder_columns(df):
    columns = [
        'session_id', 'site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
        'site7', 'site8', 'site9', 'site10', 'target', 
        'time1_year', 'time1_month', 'time1_day', 'time1_hour', 'time1_minute', 'time1_second',
        'time2_year', 'time2_month', 'time2_day', 'time2_hour', 'time2_minute', 'time2_second',
        'time3_year', 'time3_month', 'time3_day', 'time3_hour', 'time3_minute', 'time3_second',
        'time4_year', 'time4_month', 'time4_day', 'time4_hour', 'time4_minute', 'time4_second',
        'time5_year', 'time5_month', 'time5_day', 'time5_hour', 'time5_minute', 'time5_second',
        'time6_year', 'time6_month', 'time6_day', 'time6_hour', 'time6_minute', 'time6_second',
        'time7_year', 'time7_month', 'time7_day', 'time7_hour', 'time7_minute', 'time7_second',
        'time8_year', 'time8_month', 'time8_day', 'time8_hour', 'time8_minute', 'time8_second',
        'time9_year', 'time9_month', 'time9_day', 'time9_hour', 'time9_minute', 'time9_second',
        'time10_year', 'time10_month', 'time10_day', 'time10_hour', 'time10_minute', 'time10_second',
        'most_visited_site', 'time1_part_of_day'
    ]
    
    # Изменение порядка столбцов
    df = df[columns]

    df['session_id'] = df['session_id'].astype(str).str.extract('(\d+)')

    return df

@app.get("/Alice_log_predict/{num_of_iter}")

def Alice_log_predict( num_of_iter : int):

    percentage = 0

    for i in range (0, int(num_of_iter)):
        df = get_data_Alice()

        df = making_more_columns(df)

        site_dict_path = 'site_dic.pkl'

        df = replace_sites_with_codes(df, site_dict_path)

        df = reorder_columns(df)

        percentage += predict_it(df)
    
    percent = (percentage/num_of_iter)*100
    
    return f"For {num_of_iter} iterations, percentage is {percent}"

@app.post("/random_user_predict/{num_of_iter}")

def random_user_log_predict(num_of_iter : int):

    percentage = 0

    for i in range (0, int(num_of_iter)):
        df = get_random_user()
        
        df = making_more_columns(df)

        site_dict_path = 'site_dic.pkl'

        df = replace_sites_with_codes(df, site_dict_path)

        df = reorder_columns(df)
        percentage += predict_it(df)
    
    percent = (percentage/int(num_of_iter))*100

    return f"For {num_of_iter} iterations, percentage is {percent}"

@app.get("/")

def main():
    decision = int( input("Привет! Добро пожаловать в раздел «Это Алиса?». Дайте мне данные, и я угадаю, Алиса это или нет :) \n Введите 1 для получения случайных данных из файла с именем Alice_log \n Введите 2 для получения данных от случайного пользователя в пакете с именем other_user_log \n Введите 3 для самостоятельного ввода данных."))

    if decision == 1:
        Alice_log_predict(1)

    elif decision == 2:
        random_user_log_predict(1)

    elif decision == 3:
        data = get_user_input()

        # Сохраняем их в DataFrame
        df = save_to_dataframe(data)

        df = making_more_columns(df)

        predict_it(df)

    else:
        print ('Неправильный выбор')



main()