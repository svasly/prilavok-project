"""
Модуль предобработки данных проекта прогнозирования продаж «Прилавок».

Содержит функции создания признаков, идентичные использованным при обучении модели
(model_training.ipynb), и единую входную функцию preprocess_data для DAG-инференса.
Идентичность предобработки обучения и инференса — требование корректной работы модели.

Защитные механизмы:
- MODEL_FEATURES / FEATURE_DTYPES — контракт состава, порядка и типов столбцов;
- ensure_model_dtypes() — восстановление типов после JSON-раундтрипа через XCom;
- лаги недель без продаж заполняются типичным уровнем пары (avg_sales_before):
  нулевые лаги читались моделью как «отдел перестал продавать» (рекомендация ревью);
- пары без истории получают базовые уровни медианой своего отдела по другим
  дарксторам: система закупок ждёт прогноз по каждой строке плана (блокер ревью).
"""
import pandas as pd

# Точный список и ПОРЯДОК признаков, на которых обучена модель (после PSI-фильтра).
# Порядок строго совпадает со столбцами обучения из тетради.
MODEL_FEATURES = [
    "store", "dept", "is_holiday", "type", "size",
    "avg_sales_before", "median_sales_before",
    "sales_1week_ago", "sales_2week_ago", "sales_4week_ago",
    "mean_sales_2week", "mean_sales_4week", "mean_sales_8week", "mean_sales_12week",
]

# Календарные лаги 1–12 недель: в модель входят 1/2/4,
# остальные нужны только для скользящих окон 4/8/12.
LAG_COLUMNS = [f"sales_{i}week_ago" for i in range(1, 13)]

# Контракт типов: модель обучена именно на таких dtype.
FEATURE_DTYPES = {
    "store": "int64",
    "dept": "int64",
    "is_holiday": "bool",
    "type": "str",
    "size": "int64",
    "avg_sales_before": "float64",
    "median_sales_before": "float64",
    "sales_1week_ago": "float64",
    "sales_2week_ago": "float64",
    "sales_4week_ago": "float64",
    "mean_sales_2week": "float64",
    "mean_sales_4week": "float64",
    "mean_sales_8week": "float64",
    "mean_sales_12week": "float64",
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
    """Средние и медианные продажи по (store, dept) строго ДО даты строки: утечка исключена.

    expanding().mean()/median() дают накопительную статистику включая текущую строку,
    shift(1) сдвигает её на неделю назад. NaN цели (недели плана) при накоплении
    пропускаются, поэтому для строк плана статистики считаются только по истории.
    """
    df = df.sort_values(["store", "dept", "date"]).copy()
    g = df.groupby(["store", "dept"])["weekly_sales"]
    df["avg_sales_before"] = g.transform(lambda s: s.expanding().mean().shift(1))
    # Медиана дополняет среднее: устойчива к выбросам (праздничные пики не перекашивают базовый уровень пары)
    df["median_sales_before"] = g.transform(lambda s: s.expanding().median().shift(1))
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные лаги 1–12 недель через сдвиг дат (устойчиво к дырам сетки)."""
    df = df.copy()
    base = df[["store", "dept", "date", "weekly_sales"]]
    for lag in range(1, 13):
        lag_df = base.copy()
        lag_df["date"] = lag_df["date"] + pd.DateOffset(weeks=lag)
        lag_df = lag_df.rename(columns={"weekly_sales": f"sales_{lag}week_ago"})
        df = df.merge(lag_df, on=["store", "dept", "date"], how="left")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Скользящие средние за 2, 4, 8 и 12 недель из лаговых столбцов.

    Текущая неделя в окно не входит (защита от утечки). Расчёт из лагов, а не из
    weekly_sales, позволяет корректно обработать недели плана: к моменту расчёта
    лаги уже заполнены типичным уровнем пары там, где продаж ещё не существует.
    """
    df = df.copy()
    df["mean_sales_2week"] = df[LAG_COLUMNS[:2]].mean(axis=1)
    df["mean_sales_4week"] = df[LAG_COLUMNS[:4]].mean(axis=1)
    df["mean_sales_8week"] = df[LAG_COLUMNS[:8]].mean(axis=1)    # среднесрочная динамика
    df["mean_sales_12week"] = df[LAG_COLUMNS[:12]].mean(axis=1)  # квартальная динамика
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

    # 3. Обогащение признаками (порядок важен: базы -> лаги -> заполнение лагов -> скользящие)
    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    # Новые пары без истории: базовые уровни — медиана того же отдела по другим дарксторам
    for col in ["avg_sales_before", "median_sales_before"]:
        dept_median = df.groupby("dept")[col].transform("median")
        df[col] = df[col].fillna(dept_median)
        df[col] = df[col].fillna(df[col].median())  # страховка: отдел целиком новый
    df = create_lag_features(df)
    # Недели, для которых продаж N недель назад не существует (план, старт истории):
    # лаги = типичный уровень пары, а не ноль (рекомендация ревью)
    for col in LAG_COLUMNS:
        df[col] = df[col].fillna(df["avg_sales_before"])
    df = create_rolling_features(df)
    # Оставляем только модельные лаги (1, 2, 4); остальные были вспомогательными
    df = df.drop(columns=[c for c in LAG_COLUMNS if c not in MODEL_FEATURES])

    # 4. Контракт: строго признаки модели + служебная date (нужна DAG для фильтра
    #    строк плана и первичного ключа витрины), типы восстановлены по контракту
    df = df[["date"] + MODEL_FEATURES]
    return ensure_model_dtypes(df)