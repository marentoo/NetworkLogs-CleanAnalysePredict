import pandas as pd
import os

def load_df(df1):
        print('----------------------------------------');print('First look on data')
        print(df1.head())
        print(len(df1.index))
        print(df1.columns)
        print(df1.dtypes)
        return df1