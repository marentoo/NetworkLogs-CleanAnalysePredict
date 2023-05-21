import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.ex3_loading import load_df
from src.ex3_cleaning import clean_df
from src.ex3_analyzing import analyze
from src.ex3_prediction import predict
from src.ex3_scaling import scaling_data

# df1 = pd.read_csv('RIPE-Atlas.csv',sep=',',dtype={'result': str})
df1 = 'RIPE-Atlas-measurement-50728410.json'

## Scale - *chose type of scaling
scaler_norm = MinMaxScaler() #scale between <0,1> (for e.g. algor: KNN or NN)
# scaler_stand = StandardScaler() #scale differently ( for e.g. Logistic Regression, Linear Discriminant Analysis)


def main():
        ##loading data and merge
    df_network_logs = load_df(df1)
        
        ##cleaning and handling noise
    df_cleaned_network_logs = clean_df(df_network_logs, 'networkdata')
        
        ##PreScaling Analysis - boxplots, histograms, conf matrix... note: there are columns as parameters for boxplot - col 1 col 2
    analyze(df_cleaned_network_logs,'networkdata',  col1=['bsize'], col2=['rt'])
        
        ##Scaling data frames by chosen scaling technique
    df_scaled_network_logs = scaling_data(df_cleaned_network_logs,'network_logs', scaler_norm)

        ##Prediction - building model and predicting
    predict(df_scaled_network_logs)

    # return df_scaled_network_logs

if __name__ == "__main__":
    main()