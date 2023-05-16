#------------------------------------------------------------------------------------
        ##Standardize data - Check for inconsistent data formats and standardize them to make analysis easier.
##https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/#Why_Should_We_Use_Feature_Scaling?

def scaling_data(df, df_name, scaler):
        ##Scaling data-select numeric data to scale - Scale the numeric columns for better performance of ML algorihms
        ##???nie wiem czy nie powinniśmy wybierac dokładnych kolumn ktore on normalizuje/standaryzuje
        exclude_cols = ['chipsettime','qualitytimestamp','gpstime']
        num_cols = df.select_dtypes(include=['int64' , 'float64']).columns.difference(exclude_cols)
        #Fit and transform the data using the scaler
        df[num_cols]=scaler.fit_transform(df[num_cols])
        df.to_csv('analysis/df_scaled_{}.csv'.format(df_name), index = False)
        return df