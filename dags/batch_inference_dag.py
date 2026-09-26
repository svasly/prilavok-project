"""
DAG для batch-инференса продаж "Прилавок"
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
import pandas as pd
import numpy as np
import boto3
from catboost import CatBoostRegressor
import logging
import io
import pickle
import tempfile

# ========== Функция предобработки данных ==========
from preprocessing import preprocess_data, MODEL_FEATURES, ensure_model_dtypes

# ========== Конфигурация ==========
S3_BUCKET = Variable.get("s3_bucket_name", default_var="your-bucket-name")
S3_ACCESS_KEY = Variable.get("s3_access_key", default_var=None)
S3_SECRET_KEY = Variable.get("s3_secret_key", default_var=None)
S3_MODEL_KEY = Variable.get("s3_model_key", default_var="catboost_sales_model.pkl")
POSTGRES_CONN_ID = "postgres_sales_db"  # ID подключения в Airflow Connections
RANDOM_STATE = 42
CAT_FEATURES = ['store', 'dept', 'is_holiday', 'type']
logger = logging.getLogger(__name__)


# ========== Задачи DAG ==========
def load_data_from_postgres(**context):
    """
    Загрузка данных для инференса из Postgres.
    Загружаем таблицы plan, stores, features и исторические продажи для расчета лаговых признаков.
    Передавать большие данные через xcom_push нельзя!
    Поэтому в данной функции создайте таблицу, которую будем использовать для предобработки данных (назовем ее inference_data_temp).
    В этой таблице соберите исторические данные (до первой даты плана), а также плановые данные (столбец weekly_sales заполните null).
    Здесь же соберите признаки из всех необходимых таблиц воедино.
    После создания выведите информацию (можно через print(), можно через logging.info):
    - количество строк в полученной таблице
    - минимальная дата
    - максимальная дата
    В XCom передайте первую дату плана (по ней будем разделять все данные на обучающие и инференс).
    """
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    # История + план в одной временной таблице, признаки подтягиваем join-ом
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp;")
    create_query = """
    CREATE TABLE inference_data_temp AS
    SELECT
        s.store,
        s.dept,
        s.date,
        s.weekly_sales,
        s.is_holiday,
        f.temperature,
        f.fuel_price,
        f.factor1,
        f.factor2,
        f.factor3,
        f.factor4,
        f.factor5,
        f.cpi,
        f.unemployment,
        st.type,
        st.size
    FROM (
        SELECT store, dept, date, weekly_sales, is_holiday FROM sales
        UNION ALL
        SELECT store, dept, date, NULL AS weekly_sales, is_holiday FROM plan
    ) AS s
    LEFT JOIN features AS f
        ON f.store = s.store AND f.dept = s.dept AND f.date = s.date
    LEFT JOIN stores AS st
        ON st.store = s.store;
    """
    cursor.execute(create_query)
    conn.commit()

    # Логируем размер и границы дат полученной таблицы
    cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM inference_data_temp;")
    total_rows, min_date, max_date = cursor.fetchone()
    logging.info(f"Строк в inference_data_temp: {total_rows}")
    logging.info(f"Минимальная дата: {min_date}, максимальная дата: {max_date}")

    # Первая дата плана — граница истории и инференса; передаём через XCom
    cursor.execute("SELECT MIN(date) FROM plan;")
    plan_start_date = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return plan_start_date


def preprocess_features(**context):
    """
    Предобработка признаков для инференса.
    1. Читаем первую дату плана из XCom
    2. Читаем данные из таблицы inference_data_temp
    3. Выполняем предобработку признаков (функция preprocess_data)
    4. Оставляем только строки, относящиеся к плану (дату берем из п.1)
    5. Не забудьте удалить колонку weekly_sales (здесь она всегда будет заполнена null)
    6. Лаговые переменные для первых недель стоит заполнить нулями.
    7. Удаляем строки с оставшимися NaN
    8. Передаем датафрейм в XCom (теперь он уже небольшой)
    9. Удалим нашу временную таблицу inference_data_temp из БД
    """
    # 1. Первая дата плана из XCom
    plan_start = pd.Timestamp(context["ti"].xcom_pull(task_ids="load_data"))

    # 2. Читаем временную таблицу
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    engine = pg_hook.get_sqlalchemy_engine()
    df = pd.read_sql("SELECT * FROM inference_data_temp", con=engine)

    # 3. Предобработка, идентичная обучению (модуль preprocessing)
    df = preprocess_data(df)

    # 4. Оставляем только строки плана
    df = df[df["date"] >= plan_start]

    # 5. weekly_sales в плане всегда null; после preprocess_data колонки уже нет —
    #    оставляем защитное удаление по шаблону
    if "weekly_sales" in df.columns:
        df = df.drop(columns=["weekly_sales"])

    # 6. Лаги первых недель — нулями (основное заполнение уже в preprocess_data)
    lag_cols = ["sales_1week_ago", "sales_2week_ago", "sales_4week_ago"]
    df[lag_cols] = df[lag_cols].fillna(0)

    # 7. Строки с оставшимися NaN удаляем
    df = df.dropna()
    logging.info(f"Строк плана, готовых к инференсу: {len(df)}")

    # 9. Временная таблица больше не нужна
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp;")
    conn.commit()
    cursor.close()
    conn.close()

    # 8. Датафрейм небольшой — передаём через XCom в виде JSON
    return df.to_json()


def load_model_from_s3(**context):
    """
    Загрузка обученной модели CatBoost из S3 (Yandex Cloud) через pickle.
    Клиент уже создан для вас. Необходимо загрузить модель используя io.BytesIO().
    Сохраните модель во временный файл для передачи через XCom.
    В XCom передайте путь до модели.
    """
    s3_client = boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="ru-central1",
    )
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
    buffer = io.BytesIO(obj["Body"].read())

    # Модель не гоняем через XCom — передаём только путь к временному файлу
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(buffer.getvalue())
        model_path = tmp.name
    logging.info(f"Модель загружена из S3 во временный файл: {model_path}")
    return model_path


def run_batch_inference(**context):
    """
    Batch-инференс: применение модели к подготовленным данным.
    1. Из XCom загрузите подготовленный датафрейм для инференса и путь к обученной модели.
    2. Прочитайте датафрейм используя pandas
    3. Загрузите модель из pickle файла
    4. Выполните predict
    5. Сохраните предсказания в качестве нового столбца - predicted_weekly_sales
    6. Обработайте аномальные предсказания (замените на 0)
    7. Передайте в XCom датафрейм с предсказаниями (используйте .to_json())
    """
    # 1. Данные и путь к модели из XCom
    df_json = context["ti"].xcom_pull(task_ids="preprocess_features")
    model_path = context["ti"].xcom_pull(task_ids="load_model")

    # 2. Читаем датафрейм
    df = pd.read_json(io.StringIO(df_json))
    df["date"] = pd.to_datetime(df["date"])

    # 3. Модель из pickle-файла
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 4-5. Прогноз; типы восстанавливаем по контракту (защита от JSON-раундтрипа)
    if df.empty:
        logging.warning("Пустой план: строк для прогноза нет. Передаём пустую схему.")
        df["predicted_weekly_sales"] = pd.Series(dtype="float64")
    else:
        df["predicted_weekly_sales"] = model.predict(ensure_model_dtypes(df[MODEL_FEATURES]))

    # 6. Продаж меньше нуля не бывает
    df["predicted_weekly_sales"] = df["predicted_weekly_sales"].clip(lower=0)
    logging.info(f"Сгенерировано предсказаний: {len(df)} строк")

    # 7. Передаём результат через XCom
    return df.to_json()


def save_predictions_to_postgres(**context):
    """
    Запись результатов предсказаний в таблицу predictions в Postgres.
    1. Загрузите датафрейм с предсказаниями из XCom
    2. Подключение к БД, создание таблицы, вставка данных уже реализованы
    3. Выведите через print или logging.info количество строк итоговой таблицы и первые 5 строк.
    """
    # 1. Датафрейм с предсказаниями из XCom
    predictions_json = context["ti"].xcom_pull(task_ids="run_inference")
    predictions_df = pd.read_json(io.StringIO(predictions_json))

    # Преобразуем date в правильный формат для PostgreSQL
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date

    # Подключение к БД
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    drop_table_query = """
    DROP TABLE IF EXISTS predictions;
    """
    cursor.execute(drop_table_query)
    conn.commit()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    from psycopg2.extras import execute_values
    values = predictions_df[['store', 'dept', 'date', 'predicted_weekly_sales']].to_records(index=False).tolist()
    insert_query = """
    INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
    VALUES %s
    ON CONFLICT (store, dept, date) 
    DO UPDATE SET 
        predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
        prediction_timestamp = CURRENT_TIMESTAMP;
    """
    execute_values(cursor, insert_query, values)
    conn.commit()

    # 3. Контроль: количество строк и первые 5 строк витрины
    cursor.execute("SELECT COUNT(*) FROM predictions;")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT * FROM predictions LIMIT 5;")
    head = cursor.fetchall()
    logging.info(f"Строк в таблице predictions: {total}")
    logging.info("Первые 5 строк таблицы predictions:")
    for row in head:
        logging.info(row)
    cursor.close()
    conn.close()


# ========== Определение DAG ==========
default_args = {
    'owner': 'Василий Сысоев',
    'depends_on_past': False,
    'email': ['best_data_scientist_in_the_world@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'sales_prediction_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0',  # Каждое воскресенье в 20:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'production'],
)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)

task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)

task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)

task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions