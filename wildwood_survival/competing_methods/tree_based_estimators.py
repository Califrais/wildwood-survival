from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

def tree_based_estimators_(estimators_name, tree_params):
    """
    List tree-based estimators of competing models

    estimators_name : `str`
        estimator's name, it requires the name to be in the
        list of defined names ["SurvivalTree", "RSF", "GBSA"]

    tree_params: `dict`

    Returns
    -------
    estimators : `dict`
        Dictionary of estimators.

    """
    max_depth_ = tree_params["max_depth"]
    min_samples_leaf_ = tree_params["min_samples_leaf"]
    n_estimators_ = tree_params["n_estimators"]

    if estimators_name == "SurvivalTree":
        estimator = SurvivalTree(max_depth = max_depth_,
                                    min_samples_leaf = min_samples_leaf_)
    elif estimators_name == "RSF":
        estimator = RandomSurvivalForest(n_estimators = n_estimators_)

    elif estimators_name == "GBSA":
        estimator = GradientBoostingSurvivalAnalysis(n_estimators = n_estimators_)

    else:
        raise Exception('Can not find {} estimator in the list of supported'
                        ' names'.format(estimators_name))

    return estimator