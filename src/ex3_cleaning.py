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
def clean_miss(df, df_name):
    #check if any NaN
    sNaN = df.isnull().values.any()
    print("\n",f"Any missing data - {df_name}?: {sNaN}")

    #Number of NaN in each column
    missing_values_count = df.isnull().sum()
    print(f"Missing data count:\n{missing_values_count}\n")

    ##Handle NaN
    df['err'].fillna('no_error', inplace = True)

    #I approach - droping
    df.dropna(subset = ['rt'], inplace = True)
    df.dropna(subset = ['from'], inplace = True)
    df.dropna(subset = ['mver'], inplace = True)
    
    #II approach - imputating
    # df['ver'] = df['ver'].fillna('NaN')
    # df['mver'] = df['mver'].fillna('NaN')
    # df['rt'] = df['rt'].fillna(0).astype(float)
    # df['res'] = df['res'].fillna(0).astype(int)
    # df['hsize'] = df['hsize'].fillna(0).astype(int)
    # df['bsize'] = df['bsize'].fillna(0).astype(int)
    # df['src_addr'] = df['src_addr'].fillna('NaN')
    
    print('Handling missing data!!!',"\n")

    return df


#------------------------------------------------------------------------------------
        ##Detect outliers -  Define z_score for checking outliers and Percentage of outliers in whole dataset:
def detect_outliers(df, df_name):
    z_scores = np.abs((df - df.mean(numeric_only=True)) / df.std(numeric_only=True))
    outliers = df[(z_scores > 4).any(axis=1)]
    print(f'No. of outliers - {df_name}: {len(outliers.index)}')
    percout = int((len(outliers.index)*100)/len(df.index))
    print(f' outliers percentage- {df_name}: {percout} %\n')

    # Print the attributes considered for outlier detection
    attributes = z_scores.columns[(z_scores > 4).any(axis=0)]
    print('Attributes considered for outlier detection:')
    print(attributes)

    ##Handling outliers
    ## I approach
    df = df.drop(outliers.index)  # Drop the rows with outliers
    outliers.to_csv('analysis/outliers_dropped.csv', index=False)  # Save the cleaned DataFrame to a CSV file

    print("Dropping outliers!\n")
    print(f'Remaining rows: {len(df.index)}')
    return df

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

    ## err laczenia w jedno
    # df = df[df['err'] != 'bad chunk line: not a number']
    # replace_dict = {
    #     'timeout reading chunk: state 5 linelen 0 lineoffset 0': 'timeout reading',
    #     'timeout reading status': 'timeout reading',
    #     'timeout reading chunk: state 6 linelen 0 lineoffset 0': 'timeout reading'
    # }
    # df.loc[:, 'err'] = df['err'].replace(replace_dict)
    # df.loc[:, 'err'] = df['err'].replace(['connect: Network is unreachable'], 'connect: Network unreachable')

    df.to_csv(file_path, index = False)
    return df