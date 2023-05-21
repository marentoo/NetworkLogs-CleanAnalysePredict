import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
        ##pre analysis
def vis_bxplt(df, columns, df_name):
    df_num = df.loc[:,columns]
    _, ax = plt.subplots(figsize=(20,6))
    df_num.plot(kind = 'box', ax=ax)
    plt.savefig(f'analysis/boxplot_{df_name}.png')

#-----------------------------------------------------------------------------
        ##analyze data # create charts, planes...

def analyze(df, df_name, col1 = None, col2=None):
    #table - describe numeric columns
    stats = df.describe() #if specific attributes (columns) should be describe.
    stats.to_csv('analysis/stats_{}.csv'.format(df_name))  

    # Histograms of numeric columns
    df.hist(figsize=(20,15))
    plt.savefig('analysis/histogram_{}.png'.format(df_name))

    # Boxplots of all numeric columns
    if col1 is not None:
        vis_bxplt(df.loc[:, col1], col1, f'{df_name}_{col1}')
    if col2 is not None:
        vis_bxplt(df.loc[:, col2], col2, f'{df_name}_{col2}')

    # Correlation matrix
    corr_matrix = df.corr(numeric_only=[False/True])

    # Heatmap of correlation matrix
    plt.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.savefig('analysis/heatmap_{}.png'.format(df_name))
