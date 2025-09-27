from typing import Any, List

import numpy as np
import openml
import pandas as pd
from tabpfn import TabPFNRegressor, TabPFNClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from pandas.api.types import CategoricalDtype as CategoricalDtypeA
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtypeB
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss

import prompting

metric_func_by_task = dict(
    binary=roc_auc_score, multiclass=log_loss, regression=root_mean_squared_error
)


def get_task_variables(task: openml.OpenMLSupervisedTask, fold: int = 0, repeat: int = 0) -> dict[str, Any]:
    """Load a standard classification dataset"""
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe")
    task_type = ("multiclass" if len(y.unique()) > 2 else "binary"
                 ) if isinstance(y.dtype, (CategoricalDtypeA, CategoricalDtypeB)) else "regression"
    print(f"Inferred task type: {task_type}")
    if task_type != 'regression':
        y = LabelEncoder().fit_transform(y)
    y = np.asarray(y)
    train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y[train_indices]
    y_test = y[test_indices]
    return dict(task_type=task_type, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                categorical_indicator=categorical_indicator)


def add_tabpfn_baseline(
        *, task_type: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
        categorical_indicator: List[bool], n_splits: int = 10, kfold_random_state: int = 42) -> dict[str, Any]:
    """Modify the **outputs of get_task_variables(task) to set up stacking on top of the TabPFN baseline"""
    categorical_features = [
        col for col, is_cat in zip(X_train.columns, categorical_indicator)
        if is_cat or X_train[col].dtype is np.dtype("O")
    ]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_features), ("cat", OrdinalEncoder(), categorical_features)]
        if categorical_features else [("num", "passthrough", numeric_features)]
    )
    preprocessor.fit(pd.concat([X_train, X_test], axis=0))
    X_train_preproc = preprocessor.transform(X_train)
    X_test_preproc = preprocessor.transform(X_test)

    TabPFN = TabPFNClassifier if task_type != 'regression' else TabPFNRegressor
    predict_fn = lambda m, X: m.predict(X) if task_type == 'regression' else (
        m.predict_proba(X)[:, -1] if task_type == 'binary' else m.predict_proba(X)
    )
    tabpfn = TabPFN(
        random_state=0, model_path=None, categorical_features_indices=np.flatnonzero(categorical_indicator)
    ).fit(X_train_preproc, y_train)
    y_test_pred = predict_fn(tabpfn, X_test_preproc)
    del tabpfn
    y_train_pred = np.full((len(y_train), *y_test_pred.shape[1:]), dtype=y_test_pred.dtype, fill_value='nan')
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=kfold_random_state)
    for ids_train, ids_val in tqdm(list(kfold.split(X_train_preproc, y_train))):
        tabpfn = TabPFN(
            random_state=0, model_path=None, categorical_features_indices=np.flatnonzero(categorical_indicator)
        ).fit(X_train_preproc[sorted(ids_train)], y_train[sorted(ids_train)])
        y_train_pred[ids_val] = predict_fn(tabpfn, X_train_preproc[ids_val])
    assert not np.any(np.isnan(y_train_pred))
    del tabpfn; import torch; torch.cuda.synchronize(); torch.cuda.empty_cache()
    if task_type != 'multiclass':
        X_train['baseline_prediction'] = y_train_pred
        X_test['baseline_prediction'] = y_test_pred
    else:
        for i in range(y_train_pred.shape[1]):
            X_train[f"baseline_prediction_{i}"] = y_train_pred[:, i]
            X_test[f"baseline_prediction_{i}"] = y_test_pred[:, i]
    print(f"Baseline train {prompting.metrics_by_task[task_type]} =",
          metric_func_by_task[task_type](y_train, y_train_pred))
    print(f"Baseline test {prompting.metrics_by_task[task_type]} =",
          metric_func_by_task[task_type](y_test, y_test_pred))
    return dict(
        task_type=task_type, categorical_indicator=categorical_indicator,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        y_train_pred=y_train_pred, y_test_pred=y_test_pred,
    )