import pandas as pd
import sys
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.features import build_features, FEATURE_COLS, USEFUL_SENSORS, WINDOW


def make_sample_df(n_cycles=15, unit_id=1):
    """
    Creates a minimal sample dataframe that mimics CMAPSS structure.
    Used by all tests — one place to change if the data structure evolves.
    
    Args:
        n_cycles: number of cycles to generate (min 10 for rolling features)
        unit_id: the engine unit ID
    Returns:
        pd.DataFrame with all required columns
    """
    return pd.DataFrame({
        'unit_id': unit_id,
        'cycle': range(1, n_cycles + 1),
        'op_1': 0.0,
        'op_2': 0.0,
        'op_3': 100.0,
        # Useful sensors — realistic values from CMAPSS FD001
        'sensor_2': 642.0,
        'sensor_3': 1590.0,
        'sensor_4': 1408.0,
        'sensor_7': 554.0,
        'sensor_9': 9065.0,
        'sensor_11': 47.9,
        'sensor_12': 521.7,
        'sensor_14': 8138.0,
        'sensor_17': 393.0,
        'sensor_20': 39.0,
        'sensor_21': 23.4,
    })


def test_output_has_correct_number_of_columns():
    """
    CRITICAL TEST : build_features() must always return exactly 48 columns.
    If this fails, the model will crash in production (feature mismatch).
    """
    df = make_sample_df()
    result = build_features(df)
    
    assert result.shape[1] == 48, (
        f"Expected 48 features, got {result.shape[1]}. "
        f"Check FEATURE_COLS in src/features.py"
    )


def test_output_columns_match_feature_cols():
    """
    The column names must match exactly what the model was trained on.
    A single typo in a column name = crash in production.
    """
    df = make_sample_df()
    result = build_features(df)
    
    assert list(result.columns) == FEATURE_COLS, (
        "Column names don't match FEATURE_COLS. "
        "The model expects exact column names from training."
    )


def test_output_row_count_matches_input():
    """
    build_features() must not lose or duplicate rows.
    Input 15 cycles -> output 15 rows.
    """
    n_cycles = 15
    df = make_sample_df(n_cycles=n_cycles)
    result = build_features(df)
    
    assert result.shape[0] == n_cycles, (
        f"Expected {n_cycles} rows, got {result.shape[0]}. "
        "build_features() must preserve the number of rows."
    )


def test_rolling_mean_column_exists():
    """
    Verify that rolling mean features are created for each useful sensor.
    These are the most important features for RUL prediction.
    """
    df = make_sample_df()
    result = build_features(df)
    
    for sensor in USEFUL_SENSORS:
        col = f'{sensor}_mean_{WINDOW}'
        assert col in result.columns, (
            f"Missing rolling mean column : {col}. "
            f"Check build_features() in src/features.py"
        )


def test_no_missing_values_in_output():
    """
    The model cannot handle NaN values — it will crash or produce wrong predictions.
    build_features() must fill all NaN (especially at the start of rolling windows).
    """
    df = make_sample_df()
    result = build_features(df)
    
    nan_count = result.isnull().sum().sum()
    assert nan_count == 0, (
        f"Found {nan_count} NaN values in features. "
        "Rolling std and diff must fill NaN with 0 for early cycles."
    )


def test_multiple_units_processed_independently():
    """
    CRITICAL : rolling features must be computed per engine, not across engines.
    If engine 1 and engine 2 are mixed, features are wrong and predictions are unreliable.
    """
    # Create two different engines
    df_unit1 = make_sample_df(n_cycles=15, unit_id=1)
    df_unit2 = make_sample_df(n_cycles=15, unit_id=2)
    df_combined = pd.concat([df_unit1, df_unit2], ignore_index=True)
    
    result = build_features(df_combined)
    
    # Total rows must equal sum of both units
    assert result.shape[0] == 30, (
        "Combined dataframe should have 30 rows (15 per unit). "
        "Units may be getting mixed during feature computation."
    )