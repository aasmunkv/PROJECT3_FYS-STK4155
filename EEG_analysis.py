import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from tqdm import tqdm
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

def initializing():
    """
    Reading data, setting structure, splitting data and scaling.
    """
    data = np.array(pd.read_csv('data.csv'))[:,1:]

    X = data[:,1:-1].astype(int)
    y = data[:,-1].astype(int)
    y_binary = (y == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_binary, 
        test_size=0.25, 
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, X_test, y_train, y_test, X, y_binary)

def get_XGBmodel(depth = 5, lr = 0.08, n_est = 100):
    """
    Default input = optimal values by testing.
    """
    XGBCla = XGBClassifier(
        # Maximum depth of each tree.
        max_depth = depth,
        # Learning rate.
        learning_rate = lr, 
        # Number of trees in forest to fit.
        n_estimators=n_est, 
        verbosity=0, 
        objective='binary:logistic', 
        # Booster to use: gbtree, gblinear or dart.
        booster='gbtree', 
        # Number of parallel threads used to run xgboost.
        n_jobs=12, 
        nthread=None, 
        gamma=0, 
        min_child_weight=1, 
        max_delta_step=0, 
        # subsample: The % of rows taken to build tree. 
        # (should not be to low, recommended to be 0.8-1)
        subsample=1,
        colsample_bytree=1, 
        colsample_bylevel=1, 
        reg_alpha=0, 
        reg_lambda=1, 
        scale_pos_weight=1, 
        base_score=0.5, 
        random_state=0, 
        seed=None, 
        missing=None
    )
    return XGBCla

def finetune_n_estimators_createData():
    """
    The following code is for finetuning the n_estimators parameter in XGB.
    """
    acc, auc = [], []
    for i in tqdm([j*10 for j in range(1,31)],desc='Progress(n_estimators)',ncols=70,smoothing=0.5):
        X_train, X_test, y_train, y_test, X, y_binary = initializing()
        XGBCla = get_XGBmodel(n_est=i)
        XGBCla = XGBCla.fit(X_train, y_train)
        acc.append(accuracy_score(XGBCla.predict(X_test),y_test))
        auc.append(roc_auc_score(XGBCla.predict(X_test),y_test))
    np.save("npy-data/result_n_estimators_tuning_acc_auc_crossval_train",acc+auc)

def finetune_n_estimators(savefigure=False):
    """
    The following code loads data from above (fine tuning n_estimators), and 
    then makes plot of accuracy and ROC AUC.
    """
    res = np.array(np.load('npy-data/result_n_estimators_tuning_acc_auc_crossval_train.npy'))
    acc, auc = res[:int(len(res)/2)], res[int(len(res)/2):]
    plt.plot([j*10 for j in range(1,31)],acc,[j*10 for j in range(1,31)],auc)
    plt.xlabel('Number of trees')
    plt.ylabel('Accuracy/AUC score')
    plt.legend(["Accuracy","ROC AUC"])
    if savefigure:
        plt.savefig('img/accuracy_n_estimators_tuning_acc_auc_crossval.pdf', bbox_inches='tight')
    plt.show()

def finetune_learningrate_createData():
    """
    The following code is for finetuning the learning rate parameter for XGB.
    """
    acc,auc = [],[]
    for i in tqdm([j*0.005 for j in range(1,31)],desc='Progress(max_depth)',ncols=70,smoothing=0.5):
        X_train, X_test, y_train, y_test, X, y_binary = initializing()
        XGBCla = get_XGBmodel(lr=i)
        XGBCla = XGBCla.fit(X_train, y_train)
        acc.append(accuracy_score(XGBCla.predict(X_test),y_test))
        auc.append(roc_auc_score(XGBCla.predict(X_test),y_test))
    np.save("npy-data/result_learningrate_tuning_acc_auc_crossval_train",acc+auc)

def finetune_learningrate(savefigure=False):
    """
    The following code load accuracy (and ROC AUC) data saved from above, and 
    generates plot. Result: Learning rate looses meaningful effect when it 
    exceeds 0.05-0.1. It looks like 0.08 might be a good choice.
    """
    res = np.array(np.load('npy-data/result_learningrate_tuning_acc_auc_crossval_train.npy'))
    acc, auc = res[:int(len(res)/2)], res[int(len(res)/2):]
    plt.plot([j*0.005 for j in range(1,31)],acc,[j*0.005 for j in range(1,31)],auc)
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy/AUC score')
    plt.legend(["Accuracy","ROC AUC"])
    if savefigure:
        plt.savefig('img/xgb_accuracy_learningrate_tuning_acc_auc_crossval.pdf', bbox_inches='tight')
    plt.show()

def finetune_learningrate_gif():
    """
    The following code load accuracy data saved from above 
    (finetune_learningrate)and uses a '2n-nearest neighbour difference norm' and 
    make an animation of this result.
    """
    acc = np.load('npy-data/result_learningrate_tuning.npy')
    for n in range(len(acc)):
        acc_diff = []
        for i in range(len(acc)):
            diff_sum = 0
            for j in range(1, min([n,len(acc)-i-1])+1):
                diff_sum += (acc[i]-acc[i-j])**2
                diff_sum += (acc[i]-acc[i+j])**2
            acc_diff.append(np.sqrt(diff_sum))
        plt.figure()
        plt.plot(range(len(acc_diff)),acc_diff)
        plt.xlabel('Learning rate index')
        plt.ylabel('Accuracy difference')
        plt.savefig('img-gif/accuracy_difference'+str(n)+'.png', bbox_inches='tight')
    import imageio
    images = []
    for n in range(len(acc)):
        images.append(imageio.imread('img-gif/accuracy_difference'+str(n)+'.png'))
    imageio.mimsave('img-gif/movie_accuracy_diff.gif', images)

def finetune_depth():
    """
    The following code is for finetuning the max_depth-parameter.
    The results being: Accuracy = 0.9678 and best depth = 5 (may vary)
    """
    start_depth = 3
    tol = 10E-4
    best_depth = start_depth
    acc = [-1]
    for i in tqdm(range(20),desc='Progress(max_depth)',ncols=70,smoothing=0.5):
        XGBCla = get_XGBmodel(depth=i+start_depth)
        XGBCla.fit(X_train, y_train)
        pred = XGBCla.predict(X_test)
        acc.append(accuracy_score(y_test, pred))
        if (abs(acc[i]-acc[i+1])<tol):
            break
        if (acc[i]<acc[i+1]):
            best_depth = start_depth + i
    print("Accuracy: %.4f" % acc[-1])
    print("Best depth: %d" % best_depth)

def heatmap_createData():
    """
    Generates the set of accuracy scores for making heat map over number of 
    trees and maximum depth.
    """
    treenumber,depth = [1, 5, 10, 50, 100, 500, 1000], list(range(1,11))

    iter_list = list(product(treenumber, depth))
    result_list = np.zeros(len(iter_list))

    for i, vals in tqdm(enumerate(iter_list)):
        n, dep = vals
        model = XGBClassifier(n_estimators=n, max_depth=dep, objective="binary:logistic")
        result_list[i] = np.mean(cross_val_score(
            model, 
            X, 
            y_binary, 
            scoring='accuracy', cv=3
        ))
    np.save("npy-data/result_small_run",result_list)

def heatmap_plot(savefigure=False):
    """
    Makes heatplot of data generated and saved above (heatmap_createData).
    """
    treenumber,depth = [1, 5, 10, 50, 100, 500, 1000], list(range(1,11))
    iter_list = list(product(treenumber, depth))
    result_list = np.load('npy-data/result_big_run.npy')
    heat_arr = result_list.reshape((len(treenumber), len(depth)))

    fig, ax = plt.subplots(figsize=(8,8))
    im      = ax.matshow(heat_arr, cmap=plt.cm.coolwarm, vmin=.8, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    ax.set_xticks(np.arange(heat_arr.shape[1]))
    ax.set_xticklabels(depth)
    ax.set_yticks(np.arange(heat_arr.shape[0]))
    ax.set_yticklabels(treenumber)
    ax.set_xlim(-.5,heat_arr.shape[1]-.5)
    ax.set_ylim(heat_arr.shape[0]-.5,-.5)
    ax.set_xlabel("Maximum depth")
    ax.set_ylabel("Number of trees")
    ax.xaxis.set_ticks_position('bottom')
    plt.colorbar(im, cax=cax)
    for i, z in enumerate(heat_arr):
        for j in range(len(z)):
            ax.text(j, i, '{:.3f}'.format(z[j]), ha='center', va='center')
    if savefigure:
        plt.savefig('img/xgb_heatmap_maxdepth_n_estimators.pdf', bbox_inches='tight')
    plt.show()

def gridSearch_XGB(gridnum=3):
    """
    The following code snippet yields how to use GridSearch for parameter tune.
    It may not be suitable in the case of XGBoost due to overfitting.
    """
    n_est_list = np.array([1, 5, 10, 50, 100, 500, 1000])
    max_dep_list = list(range(1,3))
    if gridnum==1:
        grid = {'n_estimators': n_est_list, 'max_depth': np.array(max_dep_list)}
    elif gridnum==2:
        grid = {'max_depth': np.array(max_dep_list+[50])}
    else:
        grid = {'n_estimators': np.array([1000,5000])}
    XGBCla = get_XGBmodel()
    GSxgbCla = GridSearchCV(
        XGBCla, 
        grid, 
        verbose=2, 
        cv=StratifiedKFold(n_splits=5, shuffle=True)
    )
    print(GSxgbCla.best_params_)

def confidenceInterval(model, N = 30):
    """
    Creating and returning confidence intervals of accuracy and roc-auc.
    """
    predicted_accuracies = [0]*N
    predicted_roc = [0]*N
    for i in tqdm(range(N)):
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, random_state=i)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = model.fit(X_train, y_train)
        predicted_accuracies[i] = accuracy_score(model.predict(X_test), y_test)
        predicted_roc[i] = roc_auc_score(model.predict(X_test), y_test)
    r = np.mean(predicted_roc)
    m = np.mean(predicted_accuracies)

    variance_roc = np.var(predicted_roc)
    variance_acc = np.var(predicted_accuracies)
    sd_acc = np.sqrt(variance_acc)
    sd_roc = np.sqrt(variance_roc)
    CI_acc = 2*sd_acc
    CI_roc = 2*sd_roc
    return m, CI_acc, r, CI_roc

def printCI(X_train, X_test, gsearch=False):
    """
    Printing confidence intervals for KNN, RF and XGB.
    """
    if gsearch:
        #Optimal hyper paramteres for RF: n_estimators=80, criterion= "entropy", max_depth = 20
        clfKNN = KNeighborsClassifier() 
        gclfKNN = GridSearchCV(clfKNN, {"n_neighbors": [1, 2, 3, 4], 'p': [1, 2], 'weights':['distance', 'uniform'] }, verbose=3, cv=5)
        knn = gclfKNN.fit(X_train, y_train)
        print(knn.best_params_)
    else:
        #Optimal set of hyper parameters for KNN: n_neighbors=2, weights="distance", p=2
        knn = KNeighborsClassifier(n_neighbors=2, weights="distance", p=2)
    m_acc, CI_acc, m_roc, CI_roc = confidenceInterval(knn)    
    print("Confidence interval for classifier KNN: " , m_acc, "+-", CI_acc, "\n", m_roc,  "+-", CI_roc )

    clf = RandomForestClassifier(n_estimators=80, criterion= "entropy", max_depth = 20)
    m_acc, CI_acc, m_roc, CI_roc = confidenceInterval(clf)    
    print("Confidence interval for classifier RF: " , m_acc, "+-", CI_acc, "\n", m_roc,  "+-", CI_roc )

    xgb = get_XGBmodel()
    m_acc, CI_acc, m_roc, CI_roc = confidenceInterval(xgb)    
    print("Confidence interval for classifier XG: " , m_acc, "+-", CI_acc, "\n", m_roc,  "+-", CI_roc )

def compare(X_train, X_test, y_train,y_test, gsearch=False):
    """
    Comparing KNN, RF and XGB.
    """
    print("pos: ", np.sum(y_test == 1))     # counting positives in test set
    print("neg: ", np.sum(y_test == 0))     # counting negatives in test set

    # Grid search for RF
    if gsearch:
        rfc = RandomForestClassifier(random_state=0)
        params = {
                    'n_estimators': [80, 90, 100, 110],
                    'max_depth': [15, 20],
                    'criterion': ['gini', 'entropy']
                }

        clf = GridSearchCV(
            estimator=rfc,
            param_grid=params, 
            cv= 5, 
            verbose=3, 
            scoring ='accuracy'
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=80, 
            max_depth=20, 
            criterion='entropy', 
            random_state=0
        )
    model = clf.fit(X_train, y_train)
    if gsearch:
        print(clf.best_params_)
    y_pred = model.predict(X_test)
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print("ROC-AUC:",roc_auc_score(y_test, y_pred))

    # best RF model based on grid search
    clf = RandomForestClassifier(n_estimators = 80, max_depth = 20, criterion = 'entropy', random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    # for accuracy
    y_pred_rf = clf.predict_proba(X_test)[:,1]  # for roc curve

    # confusion matrix for RF
    fp = np.sum((y_pred - y_test) == 1)     # false positives
    fn = np.sum((y_test - y_pred) == 1)     # false negatives

    ind = np.where(y_pred == y_test)        # where y_pred and y_test are equal
    tp = np.sum(y_test[ind] == 1)           # true positives
    tn = np.sum(y_test[ind] == 0)           # true negatives

    print("RF tp: ", tp, " fp: ", fp, " fn: ", fn, " tn: ", tn)


    # best KNN model based on grid search
    knn = KNeighborsClassifier(n_neighbors = 2, weights="uniform", p=2)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)    # for accuracy
    y_pred_knn = knn.predict_proba(X_test)[:,1]     # for roc curve

    # confusion matrix for KNN
    fp = np.sum((y_pred - y_test) == 1)
    fn = np.sum((y_test - y_pred) == 1)

    ind = np.where(y_pred == y_test)
    tp = np.sum(y_test[ind] == 1)
    tn = np.sum(y_test[ind] == 0)

    print("KNN tp: ", tp, " fp: ", fp, " fn: ", fn, " tn: ", tn)


    # best XGB model based on finetuning
    xgb = XGBClassifier(
            max_depth = 5,
            learning_rate = 0.08,
            n_estimators=100,
            verbosity=0,
            random_state=0,
    )


    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)    # for accuracy
    y_pred_xgb = xgb.predict_proba(X_test)[:,1]     # for roc curve

    # confusion matrix for XGB
    fp = np.sum((y_pred - y_test) == 1)
    fn = np.sum((y_test - y_pred) == 1)

    ind = np.where(y_pred == y_test)
    tp = np.sum(y_test[ind] == 1)
    tn = np.sum(y_test[ind] == 0)

    print("XG tp: ", tp, " fp: ", fp, " fn: ", fn, " tn: ", tn)


    # plotting roc curves
    fpr1, tpr1, _ = roc_curve(y_test, y_pred_rf)
    fpr2, tpr2, _2 = roc_curve(y_test, y_pred_knn)
    fpr3, tpr3, _ = roc_curve(y_test, y_pred_xgb)


    plt.plot(fpr1,tpr1, label = 'RF')
    plt.plot(fpr3,tpr3, label = 'XG')
    plt.plot(fpr2,tpr2, 'o', label = 'KNN')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

if __name__=='__main__':
    """
    A small console menu for testing.
    """
    X_train, X_test, y_train, y_test, X, y_binary = initializing()
    while True:
        print("--------------------------------------------------------------")
        print("|                       Menu                       | Runtime |")
        print("|--------------------------------------------------+---------|")
        print("| 1) Compare true/false              (print/plot)  | Medium  |")
        print("| 2) Finetune n_estimators           (create data) | Long    |")
        print("| 3) Finetune n_estimators           (plot)        | Short   |")
        print("| 4) Finetune learningrate           (create data) | Long    |")
        print("| 5) Finetune learningrate           (plot)        | Short   |")
        print("| 6) Finetune max_depth              (print)       | Short   |")
        print("| 7) Heatmap n_estimators/max_depth  (create data) | Long    |")
        print("| 8) Heatmap                         (plot)        | Short   |")
        print("| 9) Confidence interval acc/roc-auc (print)       | Long    |")
        print("| Q) Quit                                          |         |")
        print("--------------------------------------------------------------")
        inp = input("Choose program to run: ")
        if inp=="1":
            compare(X_train, X_test, y_train, y_test)
        elif inp=="2":
            finetune_n_estimators_createData()
        elif inp=="3":
            finetune_n_estimators()
        elif inp=="4":
            finetune_learningrate_createData()
        elif inp=="5":
            finetune_learningrate()
        elif inp=="6":
            finetune_depth()
        elif inp=="7":
            heatmap_createData()
        elif inp=="8":
            heatmap_plot()
        elif inp=="9":
            printCI(X_train, X_test)
        elif inp in ["Q","q","quit","Quit","QUIT"]:
            break