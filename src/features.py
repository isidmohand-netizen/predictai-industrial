import pandas as pd

# Sensors identified as useful during exploration (week 1)
# Constant sensors excluded — variance < 0.01
USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_14', 'sensor_17', 'sensor_20', 'sensor_21'
]

# Rolling window size — how many cycles we look back
WINDOW = 10

# Final feature columns — must match exactly what the model was trained on
FEATURE_COLS = (
    ['cycle', 'op_1', 'op_2', 'op_3'] +
    USEFUL_SENSORS +
    [f'{s}_mean_{WINDOW}' for s in USEFUL_SENSORS] +
    [f'{s}_std_{WINDOW}' for s in USEFUL_SENSORS] +
    [f'{s}_diff' for s in USEFUL_SENSORS]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds all features from raw sensor data.
    Must produce exactly the same features as the notebook.

    Args:
        df: raw dataframe with unit_id, cycle, op_1-3, sensor_1-21
    Returns:
        df_features: dataframe with all 48 features ready for prediction
    """
    df_features = df.copy()

    for sensor in USEFUL_SENSORS:

        # Rolling mean — smoothed trend
        df_features[f'{sensor}_mean_{WINDOW}'] = (
            df_features.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        )

        # Rolling std — instability
        df_features[f'{sensor}_std_{WINDOW}'] = (
            df_features.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
        )

        # Diff — rate of change
        df_features[f'{sensor}_diff'] = (
            df_features.groupby('unit_id')[sensor]
            .transform(lambda x: x.diff().fillna(0))
        )

    return df_features[FEATURE_COLS]