#import the needed libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import csv
import pickle
import logging
import random
import sklearn
import argparse
import skopt
from skopt.callbacks import DeltaXStopper

from time import time
from glob import glob

#from scipy.sparse import csr_matrix

from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve,ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

print("sklearn version:")
print(sklearn.__version__)
print("xgb version:")
print(xgb.__version__)
logging.shutdown()

def train_xgboost_fr_skopt():

    ########################
    #command line arguments#
    ########################

    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--outdir",help="Full path to the output directory.",type=str)
    parser.add_argument("--allvars",help="Full path to the text file containing names of all variables used in the model.",type=str)
    parser.add_argument("--targetvar",help="Target variable for prediction (default=COVIDVax).",type=str,default='COVIDVax')
    parser.add_argument("--nproc",help="Number of parallel processes used default=2).",type=int,default=2)
    parser.add_argument("--varname",help="Variable name.",type=str,default='var')
    parser.add_argument("--trainfile",help="Full path to the file containing training samples.",type=str)
    parser.add_argument("--testfile",help="Full path to the file containing test samples.",type=str)
    parser.add_argument("--niter",help="Number of hyperparameter combinatons sampled for each CV run (default=75).",type=int,default=75)
    parser.add_argument("--tree_method",help="Default = hist.",type=str,default="hist",choices=['hist','gpu_hist'])
    parser.add_argument("--n_estimators",help="List for input for n_estimators xgboost paramemter.",
                        type=int,default=[100,300,800],nargs='+')
    parser.add_argument("--max_depth",help="List for input for max_depth xgboost parameter.",
                        type=int,default=[3, 5, 6, 10],nargs='+')
    parser.add_argument("--subsample",help="List of options for subsample xgboost parameter.",type=float,default=np.arange(0.5,0.95,20),nargs='+')
    parser.add_argument("--colsample_bytree",help="List of options for colsample_bytree xgboost parameter.",type=float,default=np.arange(0.5,0.95,20),nargs='+')
    parser.add_argument("--balanced",help="1 if balanced class weights are used (=default), 0 if not.",type=int,choices=[1,0],default=1)
    parser.add_argument("--seed",help="Random number generator seed (default=123).",type=int,default=123)
   
    args = parser.parse_args()
    
    #set parameters for the run
    seed = args.seed

    start = time()

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=args.outdir+args.varname+'-xgbrun.log',level=logging.INFO,filemode='w')
    logging.info("PACKAGE VERSIONS:")
    logging.info("sklearn: "+str(sklearn.__version__))
    logging.info("xgboost: "+str(xgb.__version__))
    logging.info("pandas: "+str(pd.__version__))
    logging.info("numpy: "+str(np.__version__))
    logging.info("matplotlib: "+str(matplotlib.__version__))
    logging.info("skopt: "+str(skopt.__version__))
    logging.info("csv: "+str(csv.__version__))
    logging.info("logging: "+str(logging.__version__))

    logging.info("INFO ON THE PARAMETERS AND FILE PATHS OF THIS RUN:")
    logging.info("training data: "+args.trainfile)
    logging.info("test data: "+args.testfile)
    logging.info("output directory: "+args.outdir)
    logging.info("file containing the variable groups used by the model: "+args.allvars)
    logging.info("model name: "+args.varname)

    #Hyperparameter grid for XGboost
    params = { 'max_depth': args.max_depth,
           'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
           'subsample': args.subsample,
           'colsample_bytree': args.colsample_bytree,
           #'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': args.n_estimators,
          'gamma': np.linspace(0,15,20),#[i for i in range(0,16)],
            'reg_alpha': [0],#[0,0.001,0.01,0.1,1,10,100],
            'reg_lambda': np.linspace(1,20,10)}#[1,10,100]}
    logging.info("Hyperparameter grid for XGboost:")
    logging.info(str(params))

    logging.info(str(args.niter)+" samples drawn from the grid per model.")
    logging.info("ANALYSIS STARTS")

    #read in the variables used in the current model
    print("Processing "+args.varname)
    with open(args.allvars,'rt') as infile:
        r = csv.reader(infile,delimiter=',')
        for row in r: all_vars = row
        
    #read in the training data for the current model
    df_train = pd.read_csv(args.trainfile,usecols=all_vars+[args.targetvar],na_values=['',' ','NA'])
    print("Dataframe read in...")
    print("Total number of NAs: "+str(df_train.isna().sum().sum()))
    
    #transform training set to xgboost compatible format
    class1_count = len(df_train.loc[df_train[args.targetvar]==1])
    class0_count = len(df_train.loc[df_train[args.targetvar]==0])
    X_train, y_train =  df_train.drop(args.targetvar,axis=1).values, df_train.loc[:,args.targetvar].values
    del(df_train)
    
    #compute class weights
    ratio = float(class0_count)/class1_count
    
    logging.info(args.varname+" training set read in.")
    #initialize model and hyperparameter grid
    #determine whether calss weights are used or not
    if args.balanced==1: xgb_model = xgb.XGBClassifier(objective="binary:logistic", seed=args.seed,use_label_encoder=False,eval_metric='logloss',**{"tree_method": args.tree_method}, scale_pos_weight=ratio,n_jobs=args.nproc)
    else: xgb_model = xgb.XGBClassifier(objective="binary:logistic", seed=args.seed,use_label_encoder=False,eval_metric='logloss',**{"tree_method": args.tree_method},n_jobs=args.nproc)
    clf = skopt.BayesSearchCV(estimator=xgb_model, search_spaces=params, cv=5, n_iter=args.niter, random_state=args.seed, verbose=2, n_jobs=1,n_points=2)
    logging.info(args.varname+" XGB model initialized.")
    #fit the model
    clf.fit(X_train,y_train,callback=DeltaXStopper(1e-8))
    del(X_train)
    del(y_train)
    logging.info(args.varname+" XGB model trained.")
    pickle.dump(clf,open(args.outdir+args.varname+'BayesSearchCV.pkl','wb'))
    #save the hyperparameter optimization paths to file
    cv_res_df = pd.DataFrame.from_dict(clf.cv_results_)
    cv_res_df.to_csv(args.outdir+args.varname+'optimization_path.csv',index=False)
    #read in test data and make predictions
    df_test = pd.read_csv(args.testfile,usecols=all_vars+[args.targetvar])
    #transform test set to xgboost compatible format
    X_test, y_test =  df_test.drop(args.targetvar,axis=1).values, df_test.loc[:,args.targetvar].values
    #X_test = csr_matrix(X_test)
    logging.info(args.varname+" test set read in.")
    model = clf.best_estimator_
    #save best model to file
    pickle.dump(model,open(args.outdir+args.varname+"_best_xgb_model.pkl",'wb'))
    #make predictions
    y_pred = model.predict_proba(X_test)
    del(X_test)
    #add predictions to df_test and save to file
    df_test['xgb_pred_proba'] = y_pred[:,np.where(model.classes_==1)].flatten()
    df_test.to_csv(args.outdir+args.varname+"_test_set_xgb_pred_probas.csv.gz",compression="gzip")
    del(df_test)
    logging.info(args.varname+" predictions saved to a file.")
    #draw N random subsamples of test set and compute the metrics for each subsample
    N = 10
    f = 0.75
    all_inds = [i for i in range(0,len(y_test))]
    ind_samples = []
    for i in range(0,N): ind_samples.append(random.sample(all_inds,int(f*len(all_inds))))
    #first plot precision-recall curve
    auprcs = []
    rand_AUprc = round(np.sum(y_test)/len(y_test),3)
    plt.plot(np.linspace(0,1),rand_AUprc*np.ones(shape=(1,50)).flatten(),'--k',label="random, auPRC="+str(rand_AUprc))
    for inds in ind_samples:
        auprcs.append(average_precision_score(y_test[inds],y_pred[inds,np.where(model.classes_==1)].flatten()))
        precision,recall,threshold = precision_recall_curve(y_test[inds],y_pred[inds,np.where(model.classes_==1)].flatten())
        
    if len(auprcs)==len(ind_samples): plt.plot(recall,precision,linewidth=1,c='b',label="XGBoost, AUprc="+str(round(np.mean(auprcs),3))+" ± "+str(round(np.std(auprcs),3)))
    else: plt.plot(recall,precision,linewidth=1,c='b')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir+args.varname+"_xgb_precision_recall_curve.png",dpi=300)
    plt.clf()
    logging.info(args.varname+" pr-curves computed.")
    #receiver operator characteristics curve and AUC
    aucs = []
    plt.plot(np.linspace(0,1),np.linspace(0,1),'--k',label="random, AUC=0.5")
    for inds in ind_samples:
        aucs.append(roc_auc_score(y_test[inds],y_pred[inds,np.where(model.classes_==1)].flatten()))
        fpr,tpr,threshold = roc_curve(y_test[inds],y_pred[inds,np.where(model.classes_==1)].flatten())
    
    if len(aucs)==len(ind_samples): plt.plot(fpr,tpr,linewidth=1,c='b',label="XGBoost, AUC="+str(round(np.mean(aucs),3))+" ± "+str(round(np.std(aucs),3)))
    else: plt.plot(fpr,tpr,linewidth=1,c='b')

    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir+args.varname+"_xgb_roc_curve.png",dpi=300)
    plt.clf()
    logging.info(args.varname+" roc-curves computed.")
        
    #save AUPRC and AUC values to a file
    with open(args.outdir+args.varname+"_xgb_AUPRC_AUC.txt",'wt') as outfile:
        w = csv.writer(outfile,delimiter=',')
        w.writerow(['sample','AUPRC','AUC'])
        for i in range(1,len(aucs)+1): w.writerow([i,auprcs[i-1],aucs[i-1]])
                
    #estimate confidence intervals for AUC and AUprc
    n_bootstraps = 2000
    rng_seed = 42  # control reproducibility
    bootstrapped_AUCs = []
    bootstrapped_AUprcs = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred[:,np.where(model.classes_==1)].flatten()), len(y_pred[:,np.where(model.classes_==1)].flatten()))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_test[indices],y_pred[indices,np.where(model.classes_==1)].flatten())
        bootstrapped_AUCs.append(score)
        score = average_precision_score(y_test[indices],y_pred[indices,np.where(model.classes_==1)].flatten())
        bootstrapped_AUprcs.append(score)
    
    sorted_AUCs = np.array(bootstrapped_AUCs)
    sorted_AUCs.sort()

    sorted_AUprcs = np.array(bootstrapped_AUprcs)
    sorted_AUprcs.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower_AUC = sorted_AUCs[int(0.025 * len(sorted_AUCs))]
    confidence_upper_AUC = sorted_AUCs[int(0.975 * len(sorted_AUCs))]
    mean_AUC = np.mean(sorted_AUCs)
    confidence_lower_AUprc = sorted_AUprcs[int(0.025 * len(sorted_AUprcs))]
    confidence_upper_AUprc = sorted_AUprcs[int(0.975 * len(sorted_AUprcs))]
    mean_AUprc = np.mean(sorted_AUprcs)

    #save the confidence intervals to a file
    with open(args.outdir+args.varname+"_xgb_AUPRC_AUC_CIs.txt",'wt') as outfile:
        w = csv.writer(outfile,delimiter=',')
        w.writerow(['name','AUPRC','AUC'])
        w.writerow(['mean',mean_AUprc,mean_AUC])
        w.writerow(['lower_CI',confidence_lower_AUprc,confidence_lower_AUC])
        w.writerow(['upper_CI',confidence_upper_AUprc,confidence_upper_AUC])
            
    logging.info(args.varname+" analysis completed.")    
    end = time()
    print("duration: "+str(end-start)+" s")
    
train_xgboost_fr_skopt()
