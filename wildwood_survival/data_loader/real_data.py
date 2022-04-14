import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/4.0/Resources"
from rpy2 import robjects
import pandas as pd
import numpy as np
print(os.getcwd())
robjects.r.source(os.getcwd() + "/utils/load_PBC_Seq.R")

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder

def PBC_Seq():
    """
    Load the PBC_Seq dataset

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix.

    y : structured array, shape = (n_samples, )
        A structured array containing the simulated censoring indicator as first
         field, and the simulated censored times of the event of interest
         as second field.

    """
    time_indep_feat = ['drug', 'age', 'sex']
    data_R = robjects.r["load"]()
    # TODO: encoder and normalize
    data = pd.DataFrame(data_R).T
    data.columns = data_R.colnames
    data = data[time_indep_feat + ["T_survival", "delta"]].drop_duplicates()

    n_samples = data.shape[0]
    y = np.zeros(n_samples, dtype={'names':('indicator', 'time'), 'formats':('?', 'f8')})
    y['indicator'] = data.delta
    y['time'] = data.T_survival
    X = data[time_indep_feat]

    return X, y

def GBSG():
    """
    Load the German Breast Cancer Study Group 2 dataset

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix.

    y : structured array, shape = (n_samples, )
        A structured array containing the simulated censoring indicator as first
         field, and the simulated censored times of the event of interest
         as second field.
    """
    X_tmp, y_tmp = load_gbsg2()
    X_tmp.loc[:, "tgrade"] = X_tmp.loc[:, "tgrade"].map(len).astype(int)
    X = OneHotEncoder().fit_transform(X_tmp)
    y = np.array(y_tmp.tolist(),
                 dtype=[('indicator', '?'), ('time', 'f8')])
    return X, y