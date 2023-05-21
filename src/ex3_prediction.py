""" *Note
On the bottom there is function predict() - chose function you want to call for regression or classification - depending on target - predicted value
then chose which model - parameter
uncoment what you want to initialize and comment other functions """

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os

#------------------------------------------------------------------------------------
models_reg= (
    DecisionTreeRegressor(random_state = 42, max_depth=10),
    #SVR(kernel="linear"), #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’) Dlugo?!
    GradientBoostingRegressor(random_state=42, n_estimators=60, max_depth=3, learning_rate=0.1),
    RandomForestRegressor(n_estimators=60, random_state=42, max_depth=10, max_features=0.5),
    LinearRegression()
    #Ridge(alpha=0.5),
    #Lasso(alpha=0.5)
    )

def prediction_regression(model, data, columns_feature, target, dataset_name,predictors = None):
        directory = 'evaluation'
        if not os.path.exists(directory):
                os.makedirs(directory)
        file_path_ev = os.path.join(directory, f'{type(model).__name__}_plots_{dataset_name}_{target}.png')   
                
                #load data
        #select attributes to use as predictors if none then all that are fed into
        if predictors is not None:
                data = data[predictors + [target]]

        data = pd.get_dummies(data, columns = columns_feature)
        print(data.columns)
        X = data.drop([target], axis=1)
        print(X.columns)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)

                #build model
        model = model

                #cross validation
        ## Apply cross-validation
        # scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        # rmse_scores = np.sqrt(-scores)
        # r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

        # # Print cross-validation results
        # print('---------------', type(model).__name__, '------------------')
        # print(f'Cross-validation RMSE scores: {rmse_scores}')
        # print(f'Mean RMSE: {rmse_scores.mean()}')
        # print(f'Standard deviation of RMSE: {rmse_scores.std()}')
        # print(f'Cross-validation R-squared scores: {r2_scores}')
        # print(f'Mean R-squared: {r2_scores.mean()}')
        # print(f'Standard deviation of R-squared: {r2_scores.std()}')

        # fit the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = model.predict(X_test)

                # evaluate
        # compute mean squared error
        mse = mean_squared_error(y_test, y_pred)
        # compute root mean squared error
        rmse = mse ** 0.5
        # compute R-squared
        r2 = r2_score(y_test, y_pred)

        print('---------------',type(model).__name__,'------------------')
        print(X.shape[1])
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)

        #plot the scores
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        fig.savefig(file_path_ev)
        plt.close(fig)


#-------------------------------------------------------------------------------------------------
models_clas = (DecisionTreeClassifier(random_state = 42, max_depth=10),
               GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=10, random_state=42),
               RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42),
               SVC(kernel="linear", C=0.1) #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’

)

def prediction_classification(model, data, columns_feature, target, dataset_name, predictors=None):
        directory = 'evaluation'
        if not os.path.exists(directory):
                os.makedirs(directory)
        file_path_ev2 = os.path.join(directory, f'{type(model).__name__}_ConfusionMatrix_{dataset_name}_{target}_.png')
                # Load data
        #select attributes to use as predictors if none then all that are fed into
        if predictors is not None:
              data = data[predictors + [target]]
              
        data = pd.get_dummies(data, columns=columns_feature)
        X = data.drop([target], axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Build model
        model = model
        # # Fit the model on the training data using cross-validation
        # cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        # #Print cross-validation results
        # print('---------------', type(model).__name__, '------------------')
        # print(f'Cross-validation scores: {cv_scores}')
        # print(f'Mean cross-validation score: {cv_scores.mean()}')
        # print(f'Standard deviation of cross-validation score: {cv_scores.std()}')

        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)
        
        encoder = LabelBinarizer()
        y_test_one_hot = encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))

                #Evaluate
        #Compute accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print('---------------',type(model).__name__,'------------------')
        print(X.shape)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
        
        #Plot the confusion matrix
#...
        labels=['connect: Connection refused',
                                                'connect: Host is unreachable',
                                                'connect: Network unreachable',
                                                'connect: No route to host',
                                                'connect: timeout',
                                                'no_error',
                                                'timeout reading']
        plt.figure(figsize=(14, 12))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=60)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, rotation=0)
        plt.tight_layout()
        plt.savefig(file_path_ev2)

        #plot ROC curve
#...
        n_classes = len(np.unique(y_test))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range (n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
                roc_auc[i]= auc(fpr[i], tpr[i])

        plt.figure(figsize=(8,6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan']
        for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves')
        plt.legend(loc="lower right")
        plt.savefig(f'evaluation/{type(model).__name__}_ROC_curve_{dataset_name}_{target}.png')

        # precision recall curve ...


#-------------------------------------------------------------------------------------------------
def predict(df):
        # _predictors = ['bsize','uri','from','stored_timestamp','lts']

        # prediction_regression(models_reg[0], df, ['mver','result','dst_addr','src_addr','method', 'uri', 'msm_name','type','from','timestamp'],'lts','network_logs')


        prediction_classification(models_clas[2], df, [
                'mver','dst_addr','src_addr','ver','method', 'uri',
         'msm_name','type','from','timestamp'],'err','network_logs')

