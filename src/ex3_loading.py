import pandas as pd
import json

def load_df(df1):

        with open(df1) as file:
                json_str = file.read()
        json_obj = json.loads(json_str)

        df1_og = pd.DataFrame(json_obj)
        
        df1_norm = pd.json_normalize(json_obj, 'result', meta_prefix='meta_')
        df1_og = df1_og.drop('result', axis=1)  # Drop the "result" column
        df1 = pd.concat([df1_norm, df1_og], axis=1)
        df1.to_csv('RIPE-Atlas-measurement-50728410.csv', index = False)
        
        print('----------------------------------------');print('loading and first look on data')
        print(df1.head())
        print(len(df1.index))
        print(df1.dtypes)
        return df1