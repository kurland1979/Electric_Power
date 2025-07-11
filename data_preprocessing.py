import pandas as pd
import os

def get_file(verbose=False):
    """
    Loads the household power consumption dataset from a local file.

    Parameters:
            verbose (bool): If True, prints dataset info, description, nulls, types, shape, and head.

    Returns:
            pd.DataFrame or None: The loaded dataset as a DataFrame, or None if the file is not found or loading fails.

    Notes:
            The file path is hardcoded. Expects a semicolon-separated .txt file with '?' as NaN.
    """
    
    file_path = r'C:\Users\User\OneDrive\מסמכים\Electric_Power\household_power_consumption.txt'
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, sep=';',na_values='?',low_memory=False)
        if verbose:
            print(df.info())
            print(df.describe())
            print(df.isnull().sum())
            print(df.dtypes)
            print(df.shape)
            print(df.head())
    
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def care_date_time(df,verbose=False):
    """
    Combines 'Date' and 'Time' columns into a single 'Datetime' column in datetime64 format.
    Drops the original 'Date' and 'Time' columns.
    Sorts the DataFrame by 'Datetime'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Date' and 'Time' columns.
        verbose (bool): If True, prints missing dates, head/tail of 'Datetime', and NaN count.

    Returns:
        pd.DataFrame: Modified DataFrame with a sorted 'Datetime' column.
    """
    
    df['Datetime'] = df['Date'] + ' ' + df['Time']
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df.drop(columns=['Date','Time'],inplace=True)
    
    if verbose:
        expected_range = pd.date_range(df['Datetime'].min(), df['Datetime'].max(), freq='D')
        missing_dates = expected_range.difference(df['Datetime'])
        print("Missing dates:", missing_dates)
        
    df.sort_values(by='Datetime',inplace=True)
    if verbose:
        print(df['Datetime'].head(10))
        print(df['Datetime'].tail(10))
        print(df['Datetime'].isna().sum())
  
    return df

def conversions(df):
    """
    Converts all columns except 'Datetime' to numeric type, coercing errors to NaN.
    Drops rows with any NaN values after conversion.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Datetime' column.

    Returns:
        pd.DataFrame: DataFrame with only numeric values (except 'Datetime'), no NaNs.
    """
    
    numeric_cols = df.columns.drop('Datetime') 
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    return df
        
def data_pipeline():
    """
    Run complete data preprocessing pipeline.
    
    Returns:
    - Clean DataFrame ready for modeling
    """
    df = get_file(verbose=False)
    if df is None:
        return None
        
    df = care_date_time(df, verbose=False)
    if df is None:
        return None
    
    df = conversions(df)
    if df is None:
        return None
    
    return df


























