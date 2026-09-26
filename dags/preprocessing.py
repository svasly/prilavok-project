"""
Модуль предобработки данных проекта прогнозирования продаж «Прилавок».

Содержит функции создания признаков, идентичные использованным при обучении модели
(model_training.ipynb), и единую входную функцию preprocess_data для DAG-инференса.
Идентичность предобработки обучения и инференса — требование корректной работы модели.

Защитные механизмы:
- MODEL_FEATURES / FEATURE_DTYPES — контракт состава, порядка и типов столбцов;
- ensure_model_dtypes() — восстановление типов после JSON-раундтрипа через XCom;
- заполнение лагов нулями ДО скользящих средних — иначе недели плана с 5-й onward
  получили бы NaN в скользящих и были бы съедены dropna() в DAG.
"""
import pandas as pd

# Точный список и ПОРЯДОК признаков, на которых обучена модель (после PSI-фильтра).
# CatBoost сверяет имена признаков при predict, поэтому состав и порядок критичны.
MODEL_FEATURES = [
    "store", "dept", "is_holiday", "type", "size",
    "avg_sales_before",
    "sales_1week_ago", "sales_2week_ago", "sales_4week_ago",
    "mean_sales_2week", "mean_sales_4week",
]

# Лаги, включая вспомогательный lag3 для честного 4-недельного окна
LAG_COLUMNS = ["sales_1week_ago", "sales_2week_ago", "sales_3week_ago", "sales_4week_ago"]

# Контракт типов: модель обучена именно на таких dtype.
# Восстанавливаются перед predict как защита от искажений JSON-раундтрипа через XCom.
FEATURE_DTYPES = {
    "store": "int64",
    "dept": "int64",
    "is_holiday": "bool",
    "type": "str",
    "size": "int64",
    "avg_sales_before": "float64",
    "sales_1week_ago": "float64",
    "sales_2week_ago": "float64",
    "sales_4week_ago": "float64",
    "mean_sales_2week": "float64",
    "mean_sales_4week": "float64",
}


def ensure_model_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Восстанавливает типы признаков по контракту FEATURE_DTYPES."""
    df = df.copy()
    for col, dtype in FEATURE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки из даты: месяц, квартал, год, номер недели (как при обучении)."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def create_avg_sales_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Средние продажи по (store, dept) строго ДО даты строки: утечка исключена.

    expanding().mean() даёт накопительное среднее включая текущую строку,
    shift(1) сдвигает его на неделю назад. NaN цели (недели плана) при накоплении
    пропускаются, поэтому для строк плана среднее считается только по истории.
    """
    df = df.sort_values(["store", "dept", "date"]).copy()
    df["avg_sales_before"] = (
        df.groupby(["store", "dept"])["weekly_sales"]
          .transform(lambda s: s.expanding().mean().shift(1))
    )
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги 1, 2, 3, 4 недели через календарный сдвиг дат (устойчиво к дырам сетки)."""
    df = df.copy()
    base = df[["store", "dept", "date", "weekly_sales"]]
    for lag in (1, 2, 3, 4):
        lag_df = base.copy()
        lag_df["date"] = lag_df["date"] + pd.DateOffset(weeks=lag)
        lag_df = lag_df.rename(columns={"weekly_sales": f"sales_{lag}week_ago"})
        df = df.merge(lag_df, on=["store", "dept", "date"], how="left")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Скользящие средние за 2 и 4 недели из лаговых столбцов.

    Текущая неделя в окно не входит (защита от утечки). Расчёт из лагов, а не из
    weekly_sales, позволяет корректно обработать недели плана: к моменту расчёта
    лаги уже заполнены нулями там, где продаж ещё не существует.
    """
    df = df.copy()
    df["mean_sales_2week"] = df[["sales_1week_ago", "sales_2week_ago"]].mean(axis=1)
    df["mean_sales_4week"] = df[LAG_COLUMNS].mean(axis=1)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Полный цикл подготовки данных (история + план) для инференса.

    Логика идентична обучению модели:
    1) клиппинг отрицательных продаж (возвраты) до 0;
    2) заполнение пропусков: непрерывные макропоказатели — средним,
       маркдауны — 0, категориальные — модой;
    3) обогащение признаками функциями выше;
    4) финализация контракта: столбцы date + MODEL_FEATURES, типы по контракту.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Аномальные продажи: возвраты не могут быть спросом — клиппинг до 0
    df["weekly_sales"] = df["weekly_sales"].clip(lower=0)

    # 2. Пропуски: стратегии как при обучении
    for col in ["temperature", "fuel_price", "cpi", "unemployment"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in [c for c in df.columns if c.startswith("factor")]:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    for col in ["is_holiday", "type"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Обогащение признаками (порядок важен: лаги -> нули -> скользящие)
    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    # Недели плана и старт истории: продаж "неделю назад" ещё не существует — 0 по постановке
    df[LAG_COLUMNS] = df[LAG_COLUMNS].fillna(0)
    df = create_rolling_features(df)
    df = df.drop(columns=["sales_3week_ago"])  # вспомогательный лаг, в модель не входит

    # 4. Контракт: строго признаки модели + служебная date (нужна DAG для фильтра
    #    строк плана и первичного ключа витрины), типы восстановлены по контракту
    df = df[["date"] + MODEL_FEATURES]
    return ensure_model_dtypes(df)