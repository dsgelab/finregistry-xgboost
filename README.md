# finregistry-xgboost

Code for training xgboost models using the FinRegistry matrices as input. More detailed instructions coming.

## Installation

### ePouta and FinnGen sandbox

Probably the easiest way to install the required packages is to create a separate conda environment using the environment.yml file:

``conda env create --file environment.yml``

Then activate the environment

``conda activate fr_xgb``

Note that if you use this environment in ePouta or in FinnGen sandbox. You need to import it using conda-pack (which is included in the requirements.yml). So after activating the environment, you should run

``conda pack -n fr_xgb``

More instructions regarding conda-pack in ePouta can be found from the FinRegistry master document

### Puhti



### SD Desktop

Coming soon...

## Usage instructions

See command line options by typing

```
> python train_xgboost_fr_skopt.py -h
usage: train_xgboost_fr_skopt.py [-h] [--outdir OUTDIR] [--allvars ALLVARS] [--targetvar TARGETVAR] [--nproc NPROC] [--varname VARNAME] [--trainfile TRAINFILE]
                                 [--testfile TESTFILE] [--niter NITER] [--tree_method {hist,gpu_hist}] [--n_estimators N_ESTIMATORS [N_ESTIMATORS ...]]
                                 [--max_depth MAX_DEPTH [MAX_DEPTH ...]] [--balanced {1,0}]

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Full path to the output directory.
  --allvars ALLVARS     Full path to the text file containing names of all variables used in the model.
  --targetvar TARGETVAR
                        Target variable for prediction (default=COVIDVax).
  --nproc NPROC         Number of parallel processes used default=2).
  --varname VARNAME     Variable name.
  --trainfile TRAINFILE
                        Full path to the file containing training samples.
  --testfile TESTFILE   Full path to the file containing test samples.
  --niter NITER         Number of hyperparameter combinatons sampled for each CV run (default=75).
  --tree_method {hist,gpu_hist}
                        Default = hist.
  --n_estimators N_ESTIMATORS [N_ESTIMATORS ...]
                        Comma-separated list for input for n_estimators xgboost paramemter.
  --max_depth MAX_DEPTH [MAX_DEPTH ...]
                        Comma-separated list for input for max_depth xgboost parameter.
  --balanced {1,0}      1 if balanced class weights are used (=default), 0 if not.
```
## Input

The input file format follows the finregistry-matrices data format. These matrices can be created using the code in the following repository: https://github.com/dsgelab/finregistry-matrices . Notice that all the relevant variables need to be merged into one matrix. Rows used to train the model should be in one file (--trainfile) and rows used to test in one file (--testfile). Use the flag --targetvar to set the classification target variable.

If you use the code with some other input data, make sure the data format follows what is described in the finregistry-matrices repository.

## Output

The code outputs the following files:

- ``args.outdir+args.varname+'-xgbrun.log'`` = log file
- ``args.outdir+args.varname+'BayesSearchCV.pkl'`` = skopt.BayesSearchCV-object in pickle-format
- ``args.outdir+args.varname+'optimization_path.csv'`` = optimization path from skopt
- ``args.outdir+args.varname+"_best_xgb_model.pkl"`` = xgboost model with best optimized hyperparameters in pickle-format
- ``args.outdir+args.varname+"_test_set_xgb_pred_probas.csv.gz"`` = predicted test set probabilities in gzipped csv file
- ``args.outdir+args.varname+"_xgb_precision_recall_curve.png"`` = precision-recall curve
- ``args.outdir+args.varname+"_xgb_roc_curve.png"`` = ROC-curve
- ``args.outdir+args.varname+"_xgb_AUPRC_AUC_CIs.txt"`` = AUPRC and AUC and their 95% confidence intervals estimated by bootstrapping (2000 samples).

## Notes

- Some of the hyperparameter ranges are currently hard-coded. These could be changed in the future so that they can be set using command line arguments.
- Objective is currently hard-coded as binary:logistic, should be changed if using the model in a regression task.
- Random number generator seeds are hard-coded.

