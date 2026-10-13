import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.dataframe) -> pd.DataFrame:
    """Adds technical indicators based on OHLCV columns."""
    df = df.copy()

    df['log_return'] = np.log(df['closePrice'] / df['closePrice'].shift(1))

    df['sma_10'] = df['closePrice'].rolling(window=10).mean()
    df['sma_20'] = df['closePrice'].rolling(window=20).mean()

    df['close_to_sma_10'] = df['closePrice'] / df['sma_10'] - 1
    df['close_to_sma_20'] = df['closePrice'] / df['sma_20'] - 1

    df['vol_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_momentum'] = df['volume'] / df['vol_sma_10'] - 1

    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the target variable for the model to predict."""
    df = df.copy()

    df['target_return'] = df['log_return'].shift(-1)

    return df

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main data transformation pipeline for a single ETF."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

        df = df.sort_values('date').reset_index(drop=True)

    df = add_technical_indicators(df)
    df = create_target(df)

    df = df.dropna().reset_index(drop=True)

    columns_to_drop = ['sma_10', 'sma_20', 'vol_sma_10']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df