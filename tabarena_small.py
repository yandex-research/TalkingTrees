import openml
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from tabpfn import TabPFNRegressor, TabPFNClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import root_mean_squared_error, log_loss, roc_auc_score

# Get datasets with under 2500 samples

tabarena_version = "tabarena-v0.1"
benchmark_suite = openml.study.get_suite(tabarena_version)
task_ids = benchmark_suite.tasks
small_task_ids = []

for task_id in task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    n_samples = dataset.qualities["NumberOfInstances"]
    if n_samples < 2_500:
        small_task_ids.append(task_id)
        print(dataset.name, int(n_samples), task_id)

# NOTE repeated runs are actually important with such small datasets
# Also using mutliple outer-validation folds may be important
seed = 0

df_arena_lite_results = pd.read_parquet(
    "https://tabarena.s3.us-west-2.amazonaws.com/results/df_results_lite_leaderboard.parquet"
)

for task_id in small_task_ids:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )

    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    categorical_features = [
        col
        for col, is_cat in zip(X.columns, categorical_indicator)
        if is_cat or X[col].dtype is np.dtype("O")
    ]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OrdinalEncoder(), categorical_features),
        ]
        if categorical_features
        else [("num", "passthrough", numeric_features)]
    )

    # Fit preprocessor and transform data
    preprocessor.fit(X)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    if isinstance(y_train.dtype, CategoricalDtype):
        task_type = "binclass" if len(y_train.cat.categories) == 2 else "multiclass"
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        TabPFN = TabPFNClassifier
        XGB = XGBClassifier
        Tree = DecisionTreeClassifier

        # AUC and LogLoss as in tabarena
        metric = (
            log_loss
            if task_type == "multiclass"
            else lambda gt, pred: 1 - roc_auc_score(gt, pred[:, 1])
        )
    else:
        task_type = "regression"
        TabPFN = TabPFNRegressor
        XGB = XGBRegressor
        Tree = DecisionTreeRegressor

        # This is normalization
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train = (y_train - y_train_mean) / y_train_std

        # RMSE as in tabarena
        metric = lambda gt, pred: root_mean_squared_error(
            gt, pred * y_train_std + y_train_mean
        )

    # default tabpfnv2 (strong baseline)
    tabpfn = TabPFN(
        random_state=0,
        model_path=None,
        categorical_features_indices=np.flatnonzero(categorical_indicator),
    )
    tabpfn.fit(X_train, y_train)

    # simple decision tree
    tree = Tree(random_state=0)
    tree.fit(X_train, y_train)

    # XGB (mostly default)
    xgb = XGB(enable_categorical=True, n_estimators=200, device="cuda")
    xgb.fit(X_train, y_train)

    if task_type in ["binclass", "multiclass"]:
        tabpfn_pred = tabpfn.predict_proba(X_test)
        tree_pred = tree.predict_proba(X_test)  # type: ignore
        xgb_pred = xgb.predict_proba(X_test)  # type: ignore
    else:
        tabpfn_pred = tabpfn.predict(X_test)
        tree_pred = tree.predict(X_test)
        xgb_pred = xgb.predict(X_test)

    tabpfn_score = metric(y_test, tabpfn_pred)
    tree_score = metric(y_test, tree_pred)
    xgb_score = metric(y_test, xgb_pred)

    df_cur_dataset_results: pd.DataFrame = df_arena_lite_results[  # type: ignore
        (df_arena_lite_results.dataset == dataset.name)
        & (
            df_arena_lite_results.method.isin(
                [
                    "TABPFNV2 (default)",
                    "TABPFNV2 (tuned)",
                    "XGB (default)",
                    "XGB (tuned)",
                    "TABM (tuned)",
                ]
            )
        )
    ]

    loss_name = {
        "regression": "RMSE",
        "binclass": "1 - AUC",
        "multiclass": "logloss",
    }

    print()
    print("-" * 100)
    print()

    print(f"Results on {dataset.name} {loss_name[task_type]}")

    print("\n ==== Arena Results ====")
    print(
        df_cur_dataset_results[["method", "metric_error"]]  # type: ignore
        .sort_values(by=["metric_error"])
        .reset_index(drop=True)
    )
    print("\n\n ==== Our scores ====")
    print(f"{'TabPFN':<10} {tabpfn_score:.3f}")
    print(f"{'XGB':<10} {xgb_score:.3f}")
    print(f"{'Tree':<10} {tree_score:.3f}")
