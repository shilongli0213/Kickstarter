# -*- coding: utf-8 -*-
"""
@author: Shilong

"""

# building machine learning model

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn import model_selection, metrics
#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# prepare training data
df = pd.read_pickle('Processed_features.pkl')
y_features = ['state','pledged']
x_features = list(set(list(df.columns))-set(y_features))
train_X = df[x_features]
train_y = df[y_features]

# run xgboost model
def runXGB(train_X, train_y, test_X, test_y = None, test_X2 = None, seed_val = 0, child = 1, colsample = 0.3):
    """Run XGBoost model with given data set and parameters
    
    Args:
        train_X: training data set
        train_y: training label
        test_X: testing data set
        test_y: testing label
        test_X2: future testing data
        seed_val: random seed
        child: used to set min_child_weight parameter
        colsample: used to set colsample parameter
    Returns:
        pred_test_y: predicted testing label
        pred_test_y2: predicted future testing label
        evals_result: results of eval_metrics
        model: trained xgboost model"""
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 2
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 1000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    evals_result = {}

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20, evals_result = evals_result)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds, evals_result = evals_result)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    else:
        pred_test_y2 = None
    return pred_test_y, pred_test_y2, evals_result, model


# Function to create confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, title='Confusion matrix', cmap = plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Args:
        cm: confusion matrix
        classes: class labels
    Returns
    No returns. Plot confusion matrix"""
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# train xgboost model only with meta features (without transfering text into vector)
train_X_meta = train_X.drop(columns = 'normalized_text')
train_y_meta = train_y['state']
kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 2018)
cv_scores = []
#pred_full_test = 0
pred_train = np.zeros((train_X_meta.shape[0], 2))
for dev_index, val_index in kf.split(train_X_meta):
    dev_X, val_X = train_X_meta.iloc[dev_index], train_X_meta.iloc[val_index]
    dev_y, val_y = train_y_meta.iloc[dev_index], train_y_meta.iloc[val_index]
    pred_val_y, pred_test_y, evals_result, model = runXGB(dev_X, dev_y, val_X, val_y, seed_val=0)
    #pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    
    epochs = len(evals_result['train']['mlogloss'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, evals_result['train']['mlogloss'], label='Train')
    ax.plot(x_axis, evals_result['test']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()
    
print("cv scores : ", cv_scores)


# plot confusion matrix
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=['failed', 'successful'],
                      title='Confusion matrix without text vector transfer')
plt.show()

