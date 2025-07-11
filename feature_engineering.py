import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def check_target(df,verbose=False):
    """
    Performs exploratory analysis on the target column 'Global_active_power'.
    Optionally prints numeric conversion, descriptive statistics, and plots the target.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Global_active_power' column.
        verbose (bool): If True, prints analysis and shows a plot.

    Returns:
        None
    """
    if verbose:
        print(pd.to_numeric(df['Global_active_power'], errors='coerce'))
        print(df['Global_active_power'].describe())
        df['Global_active_power'].plot()
        plt.show()
        
    return None

def get_season(month):
    """
    Returns the season ('winter', 'spring', 'summer', 'autumn') for a given month.

    Parameters:
        month (int): Month as integer (1-12).

    Returns:
        str: Season name.
    """
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def build_daily_consumption(df):
    """
    Aggregates power consumption data to a daily level, creates basic date features.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Datetime' column and 'Global_active_power'.

    Returns:
        pd.DataFrame: Daily aggregated DataFrame with date, month, dayofweek, is_weekend, and season.
    """
    
    df['Date'] = df['Datetime'].dt.date
    # Daily aggregation
    daily_df = df.groupby('Date')['Global_active_power'].sum().reset_index()

    # Convert column
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    # Produce daily features
    daily_df['month'] = daily_df['Date'].dt.month
    daily_df['dayofweek'] = daily_df['Date'].dt.dayofweek 
    
    daily_df['season'] = daily_df['month'].apply(get_season)
    
    return daily_df

def rolling_window(daily_df):
    """
    Adds rolling mean features (7, 14, and 30 days) to the DataFrame,
    based on shifted target values. Removes all rows with missing values.

    Parameters:
        daily_df (pd.DataFrame): DataFrame with a 'Global_active_power' column.

    Returns:
        pd.DataFrame: DataFrame with rolling features and no NaN rows.
    """
    
    daily_df['rolling_mean_3'] = daily_df['Global_active_power'].shift(1).rolling(window=3).mean()
    daily_df['rolling_mean_7'] = daily_df['Global_active_power'].shift(1).rolling(window=7).mean()
    daily_df['rolling_mean_30'] = daily_df['Global_active_power'].shift(1).rolling(window=30).mean()
    
    daily_df.dropna(inplace=True)
    
    return daily_df
    
def conversions(daily_df):
    """
    Converts 'is_weekend' to integer and encodes 'season' as numeric. Drops original 'season' column.

    Parameters:
        daily_df (pd.DataFrame): DataFrame with 'is_weekend' and 'season' columns.

    Returns:
        pd.DataFrame: Modified DataFrame with 'is_weekend' as int and 'season_encoded'.
    """
    
    
    le = LabelEncoder()
    daily_df['season_encoded'] = le.fit_transform(daily_df['season'])
    daily_df.drop(columns=['season'], inplace=True)

    
    return daily_df
    
def consumption_pipeline(df, verbose=False):
    """
    Run complete data preprocessing pipeline.
    
    Returns:
    - Drawing to create new features
    """
    if verbose:
        check_target(df, verbose=True)
    daily_df = build_daily_consumption(df)
    daily_df = rolling_window(daily_df)  
    daily_df = conversions(daily_df)
    
    return daily_df



    
   



