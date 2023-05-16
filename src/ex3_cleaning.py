import numpy as np
import os

##Cleaning data
#------------------------------------------------------------------------------------
        ##check duplicates and ##Checking percentage of duplicates and ##Removing all duplicates

def clean_dupl(df, df_name):

    duplicate_rows = df[df.duplicated()]
    print(f'No. duplicated rows - {df_name}: {len(duplicate_rows.index)}')
    percentage = int(len(duplicate_rows.index) * 100) / len(df.index)
    print(f'Duplicate percentage - {df_name}: {percentage:.2f}%')    

    df.drop_duplicates(keep='last', inplace = True)
    if len(duplicate_rows.index) != 0:
        print("Droping duplicates!!!")
    return df

#------------------------------------------------------------------------------------
        ##Checking for missing data and Handle missing data - no missing data! but if ... then:

def clean_miss(df,df_name):
    #check if any NaN
    sNaN = df.isnull().values.any()
    print("\n",f"Any missing data - {df_name}?: {sNaN}")

    #Number of NaN in each column
    missing_values_count = df.isnull().sum()
    print(f"Missing data count:\n{missing_values_count}\n")

    #Handle NaN
    df['result'].fillna('success', inplace = True)
    df.dropna(subset = ['rt'], inplace = True)
    df.dropna(subset = ['from'], inplace = True)
    print('Handling missing data!!!',"\n")

    missing_values_count = df.isnull().sum()
    print("\n",f"Missing data count:\n{missing_values_count}\n")

    return df


#------------------------------------------------------------------------------------
        ##Detect outliers -  Define z_score for checking outliers and Percentage of outliers in whole dataset:

def detect_outliers(df, df_name):
    z_scores = np.abs((df - df.mean(numeric_only=True)) / df.std(numeric_only=True))
    outliers = df[(z_scores > 4).any(axis=1)]
    print(f'No. of outliers - {df_name}: {len(outliers.index)}')
    percout = int((len(outliers.index)*100)/len(df.index))
    print(f' outliers percentage- {df_name}: {percout} %\n')
    return df

#------------------------------------------------------------------------------------
       ##Handling outliers <-- Drop outliers from the dataframes #<<-dziala ale zle <<--- pytanie gdzie umieścic
## df_downloads = df_downloads.drop(outliersD.index,)
## df_uploads = df_uploads.drop(outliersU.index)
## print("\n","Droping outliers!","\n")
## print(f'Rd:{len(df_downloads.index)}');print(f'Ru:{len(df_uploads.index)}')

#------------------------------------------------------------------------------------
        ##final function
def clean_df(df, df_name):
    print('----------------------------------------');print('Cleaning')
    directory = 'analysis'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'df_cleaned_{df_name}.csv')
    
    df = clean_dupl(df, f'{df_name}')
    df = clean_miss(df,f'{df_name}')
    df = detect_outliers(df, f'{df_name}')
    
    df.to_csv(file_path, index = False)
    return df