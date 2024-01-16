#import cProfile
import pstats
from functools import partial
import numpy as np
import pandas as pd
import timeout_decorator
from hyperopt import fmin, tpe, hp
from hyperopt import STATUS_OK
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MaxAbsScaler

from sklearnex import patch_sklearn
from hyperopt.mongoexp import Trials

np.random.seed(0)

patch_sklearn()  # Extension to speed calculation (X10-100)

# Profiler to see where the code is taking the most time
# profiler = cProfile.Profile()
# profiler.enable()


final_train = pd.read_csv("/home/onyxia/work/aml_project/data/raw_data/data_fot_ML/final_train_train.csv")
y_train = pd.read_csv("/home/onyxia/work/aml_project/data/raw_data/data_fot_ML/Y_train.csv")["y"]
trials = Trials()

# Our space of hyperparameters: scaler, feature extractor, and hyperparameters of Random Forest
space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300, 400]),
    "max_depth": hp.choice("max_depth", [None, 10, 20, 30]),
    "min_samples_split": hp.choice("min_samples_split", [2, 3, 7, 12]),
    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 3, 6, 10]),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "warm_start": hp.choice("warm_startbool", [True, False]),
    "max_features": hp.choice("max_features", ["sqrt", "log2"]),
    "preprocessor": hp.choice(
        "preprocessor", ["StandardScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"]
    ),
    "feature_extractor": hp.choice(
        "feature_extractor",
        [
            {"type": "pca", "n_components": hp.choice("n_components", [10, 50, 60, 75, 100, 150, 200]), "whiten": hp.choice("whiten", [True, False])},
            {"type": "RFE", "n_features_to_select": hp.choice("n_features_to_select", [200, 300, 400]), "step": 10},
            {"type": "SelectKBest", "k": hp.choice("k", [10, 75, 150, 200, 300, 400])},
        ],
    ),
}

def model_from_param(params, X, y):
    """Model to implement the pipeline and cross validate over a set of parameters"""
    print("Params testing: ", params)

    rf_model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        bootstrap=params["bootstrap"],
        warm_start=params["warm_start"],
        max_features=params["max_features"],
    )

    imputer = SimpleImputer(missing_values=np.nan, strategy="constant")
    var_filter = VarianceThreshold()

    preprocessor_dict = {
        "StandardScaler": StandardScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "Normalizer": Normalizer,
        "RobustScaler": RobustScaler,
    }

    scaler = preprocessor_dict[params["preprocessor"]]()

    if params["feature_extractor"]["type"] == "pca":
        opts = dict(params["feature_extractor"])
        del opts["type"]
        feature_extraction = PCA(**opts)

    elif params["feature_extractor"]["type"] == "RFE":
        opts = dict(params["feature_extractor"])
        del opts["type"]
        svr = svm.SVR(kernel="linear")
        feature_extraction = RFE(estimator=svr, **opts)

    elif params["feature_extractor"]["type"] == "SelectKBest":
        opts = dict(params["feature_extractor"])
        del opts["type"]
        feature_extraction = SelectKBest(score_func=f_regression, **opts)

    model = Pipeline(
        steps=[
            ("imputer", imputer),
            ("filter", var_filter),
            ("scaler", scaler),
            ("feature_extraction", feature_extraction),
            ("rf_model", rf_model),
        ]
    )

    @timeout_decorator.timeout(20, timeout_exception=StopIteration)
    def cross_val(model, X, y, cv):
        y_cv_predict = cross_val_predict(model, X, y, cv=KFold(n_splits=5), n_jobs=None)
        acc = mean_absolute_error(y, y_cv_predict)
        return acc

    n_repeats = 5
    acc = np.zeros(n_repeats)
    timeout = np.zeros(n_repeats)
    for i in range(n_repeats):
        try:
            acc[i] = cross_val(model, X, y, 5)
        except RuntimeError as e:
            if "generator raised StopIteration" in str(e):
                timeout[i] = 1
                print("Exception : Too much time, move on to next cross validate")
            else:
                raise

    if np.sum(timeout) == n_repeats:
        mae = 300
    else:
        mae = acc[timeout == 0].mean()

    return {"loss": mae, "status": STATUS_OK, "params": params}

if __name__ == "__main__":
    # Mettez le code que vous voulez exécuter en tant que programme principal ici
    # Par exemple, le code de l'optimisation hyperparamétrique

    trials = Trials()

    # minimization of the mae over 100 sets of parameters
    best = fmin(
        fn=partial(model_from_param, X=final_train, y=y_train),
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
    )

    print("Best estimator:", best)

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # uncomment to see the repartition of spending time across different functions
    # stats.sort_stats('cumulative').print_stats(10)

