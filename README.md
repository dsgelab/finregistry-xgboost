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

## Output

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

## Notes

- Some of the hyperparameter ranges are currently hard-coded. These could be changed in the future so that they can be set using command line arguments.
- Objective is currently hard-coded as binary:logistic, should be changed if using the model in a regression task.

