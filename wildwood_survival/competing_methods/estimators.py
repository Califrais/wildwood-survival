from time import time
from sksurv.linear_model import CoxPHSurvivalAnalysis as CoxPH
from competing_methods.tree_based_estimators import tree_based_estimators_
import numpy as np
import xgboost as xgb
from sksurv.metrics import concordance_index_censored as C_index

supported_tree_based_estimator = ["SurvivalTree", "RSF", "GBSA"]

class competing_estimators:
    def __init__(self, estimator_names, tree_params, xgboost_params=None):
        self.estimator_names_ = estimator_names
        self.tree_params_ = tree_params
        self.xgboost_params_ = xgboost_params
        self.estimators = self.loader()

    def loader(self):
        estimators = {}
        for name in self.estimator_names_:
            if name in supported_tree_based_estimator:
                estimators[name] = tree_based_estimators_(name, self.tree_params_)
            elif name == "Cox_PH":
                estimators[name] = CoxPH()
            elif name == "XGBoost":
                estimators[name] = None
            else:
                raise Exception(
                    'Can not find {} estimator in the list of supported'
                    ' names'.format(name))

        return estimators

    def fit(self, X, y, score="C_index"):
        perf_stats = {}
        for name, estimator in self.estimators.items():
            if name == "XGBoost":
                start = time()
                data = xgb.DMatrix(X)
                n_samples = len(y)
                indicator = y['indicator']
                true_times = y['time']
                y_lower_bound = np.zeros(n_samples)
                y_upper_bound = np.zeros(n_samples)
                for i in range(n_samples):
                    delta_, t_ = y[i]
                    if delta_:
                        y_lower_bound[i] = t_
                        y_upper_bound[i] = +np.inf
                    else:
                        y_lower_bound[i] = 0.0
                        y_upper_bound[i] = t_
                data.set_float_info('label_lower_bound', y_lower_bound)
                data.set_float_info('label_upper_bound', y_upper_bound)

                bst = xgb.train(self.xgboost_params_, data, num_boost_round=5,
                                evals=[(data, 'train')])
                # make prediction
                pred_times = bst.predict(data)
                running_time = time() - start
                score = C_index(indicator, true_times, pred_times)[0]
            else:
                start = time()
                fitted_estimator = estimator.fit(X, y)
                running_time = time() - start
                score = fitted_estimator.score(X, y)
            perf_stats[name] = (score, running_time)

        return perf_stats