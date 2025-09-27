editable_tree_api_reference = """
You are given a library with a customizable tree that can be viewed and edited manually or semi-automatically.

<code>
from editable_tree import Tree
# the tree can be created manually or exported (Tree.from_sklearn(...))
tree = Tree("foo <= 0.5", le=3.5, gt=Tree("bar <= 9", le=-2, gt=+2))

# The API always uses pd.DataFrame for feature representations (for named columns).
print(tree.predict(pd.DataFrame([dict(foo=0.6, bar=8)])))    # will output [-2.0]

# Equivalently, the same tree can be defined more explicitly as follows:
tree = Tree(id=0, feature='foo', threshold=0.5,           # if X['foo'] <= 0.5:
            left=Tree(id=1, value=3.5),                   #   return 3.5
            right=Tree(id=2, feature='bar', threshold=9,  # elif X['bar'] <= 9:
                       left=Tree(id=3, value=-2),         #       return -2
                       right=Tree(id=4, value=+2)))       # else: return +2
</code>

Core `Tree` methods and behaviours:
* `print(tree)` — textual diagnostic of the tree; use it before and after edits. It prints node indices for reference.
* `predict(X: pd.DataFrame) -> np.ndarray` — run the tree on input features (requires column names) and predict scores.
  - For classification, `Tree.predict(X)` returns probabilities for class "1" for 2-class classification and all class probabilities [num_samples, num_classes] for multiclass.
  - Note that the editable `Tree` does not have `.predict_proba` for classification - it returns probabilities with `.predict`.
* `node = tree.find_node(id)` — returns a subtree (node) with the given id (use ids from `print(tree)`).
* Node properties: `node.id, node.feature, node.threshold, node.left, node.right`
  - **Left child / `le`** corresponds to `feature <= threshold` (NaN values are routed to `le`).
  - **Right child / `gt`** corresponds to `feature > threshold`.
* `node.is_leaf` — True if node has no children. `node.le` is alias for `node.left`; `node.gt` is alias for `node.right`.
* `tree.get_data_indices_for_node(id: int, X: pd.DataFrame)` — returns `np.ndarray` of sample indices routed to that node. Use this to inspect data falling into a node or to fit subtrees locally.
* `tree.prune(id)` — remove all children of a given node (if `id` omitted, prunes the current node/root as appropriate).
* `Tree.from_sklearn(estimator: DecisionTreeRegressor|Classifier)` — import a trained sklearn tree. The sklearn estimator **must** have been fit with a `pd.DataFrame` (so it stores `feature_names_in_`).
* `tree.replace_subtree(id, subtree)` — graft a subtree at `id`, replacing existing children if present. If node indices conflict, call `.repair()` afterwards.
* `tree.grow_subtree(id, X, y, sample_weight=None, **sklearn_kwargs)` — train a subtree to extend a node at `id`. Use `get_data_indices_for_node(...)` to extract the X/y routed to that node before calling; otherwise the subtree will be trained on the full provided dataset. If the node was not a leaf, its previous children will be replaced.
* `tree.repair(reindex_all: bool = False)` — fill missing (`None`) node indices and remove duplicates. If `reindex_all=True`, node ids are renumbered to contiguous values (this will change existing ids).
* `tree.fill_values(X, y, sample_weight=None, recompute_all=False, fill_empty=False, prune_empty=False)` — compute and fill missing `.value` in leaves/subtrees (weighted means, etc.). If no data is present for a node, default behavior keeps `None`; use `fill_empty=True` to copy from an ancestor or `prune_empty=True` to remove empty branches.
  - Note that `node.value` can be a float value (regression or binary classification) or a numpy array (multiclass, multiregression). Use `float(node.value)` or `list(node.value)` for printing, according to the task.

Quick usage example (regression):
<code>
from copy import deepcopy
from editable_tree import Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
assert hasattr(tree, 'feature_names_in_')
original_val_mse = mean_squared_error(y_val, tree.predict(X_val))
print("Val MSE with original subtree:", original_val_mse)

# convert a trained sklearn tree into editable_tree, browse the root node properties and subtrees
tree = Tree.from_sklearn(tree)  # from_sklearn requires that the estimator was fit with a DataFrame as X
print(f"Tree depth: {tree.get_max_depth()}, size: {tree.get_num_nodes()}")
print(tree, "\\n")  # a text view of the tree with node indices, feature, threshold, left/right and value
print(f"Root node: {tree.id=}, {tree.feature=} (id), {tree.threshold=}\\n")
print(f"Left subtree, where X[:, {tree.feature}] <= {tree.threshold}:\\n{tree.left}\\n")
print(f"Right subtree, where X[:, {tree.feature}] > {tree.threshold}:\\n{tree.right}\\n")

# inspect a particular node and the rows that route to it. Note that `node` has the same methods as `tree`
node = tree.find_node(id=5)  # uses node id from print(tree); raises ValueError if not found
print(f"Node {node.id} subtree depth: {node.get_max_depth()}, subtree size: {node.get_num_nodes()}")
print(f"Node {node.id} subtree structure:\\n{node}\\n")  # prints a portion of the tree

# select subsets of X_train and X_val that correspond to this node
ids_train = tree.get_data_indices_for_node(node.id, X_train)  # use via X_train.iloc[ids_train], y_train[ids_train]
ids_val = tree.get_data_indices_for_node(node.id, X_val)      # similar for X_val, y_val
print(f"Num samples at node {node.id}: train={len(ids_train)}, val={len(ids_val)}")
# note that this method should be called from the root tree object, not from a detached node object

# train a sub-tree on the data that matches the selected node
subtree = DecisionTreeRegressor(max_depth=2).fit(X_train.iloc[ids_train], y_train[ids_train])
y_val_pred_for_subtree = subtree.predict(X_val.iloc[ids_val])
print("Val partial MSE for subtree:", mean_squared_error(y_val[ids_val], y_val_pred_for_subtree))

# graft a new sub-tree (replacing previously existing sub-tree if node wasn't a leaf)
modified_tree = deepcopy(tree)  # replacements are in-place by default
modified_tree.replace_subtree(node.id, Tree.from_sklearn(subtree))
modified_val_mse = mean_squared_error(y_val, modified_tree.predict(X_val))
print(f"Val MSE with grafted subtree: {modified_val_mse}, better? =", modified_val_mse < original_val_mse)

# prune a node from subtree, using its printed node id (e.g. to avoid overfitting or undo a bad split)
assert modified_tree.find_node(id=12).get_max_depth() != 0
modified_tree.prune(id=12)
assert modified_tree.find_node(id=12).get_max_depth() == 0
y_pred = modified_tree.predict(X_train)

# repair the resulting tree: fix id=None and re-numerate nodes (breadth-first order)
modified_tree.repair(reindex_all=True)
# compute any missing leaf and node values (as averages of y subsets), prune subtrees with zero samples
modified_tree.fill_values(X=X_train, y=y_train, sample_weight=None, recompute_all=False, prune_empty=True)
</code>

For classification tasks, `Tree` works similarly, but its `predict` method outputs class-1 probability (or a probability vector for multi-class).  
""".strip()

workflow_suggestions = f"""
Recommended iterative workflow (use this or your own intuition):
1. View the data and do necessary preprocessing.
   - Handle categorical features, missing data and obvious errors before training.
   - Encode categories (OneHot, manual ordinal scoring, etc.) consistently for train and val.
   - Apply identical transformations to `X_train` and `X_val`.
   - Deal with NaNs, filter or mark outliers (but **do not** remove outliers from the validation set).
   - Be prepared to revisit preprocessing after inspecting the tree.

2. Train a baseline tree.
   - Try a few sensible hyperparameter settings (e.g., `max_depth`, `min_samples_split`) to get an interpretable, reasonable starting tree.
   - Choose one initial tree to iterate on (by validation score and interpretability) and convert it to `editable_tree.Tree`.
   - Print full structure and per-node sample counts for nodes you plan to inspect.

3. Form explicit hypotheses about problematic splits or useful features (write them in `Thought:`).
   - Use `print(tree)` and `print(tree.find_node(id))` to inspect structure.
   - Compute statistics and distributions (means, std, histograms, `np.unique`, class balances).
   - Inspect training and validation rows that route to a node: `ids = tree.get_data_indices_for_node(id, X_val)` and then `print(X_val.iloc[ids], y_val[ids].mean())`.
   - Document hypotheses even if they turn out wrong — fail fast and keep what works.

4. Improve the tree based on observations:
   a. Manual grafting:
     - Compose replacements as nested `Tree(...)` objects: e.g. `subtree = Tree("feat1 <= 0.5", le=3.5, gt=Tree("feat2 <= 9", le=-2, gt=+2))`
     - Optionally infer leaf values automatically: `subtree.fill_values(X.iloc[ids], y[ids], recompute_all=True)`
     - Graft with: `tree.replace_subtree(node_id, subtree)`
   b. Local automatic growth:
     - Extract local indices: `ids = tree.get_data_indices_for_node(id, X_train)` and train a sklearn estimator on `X_train.iloc[ids], y_train[ids]`.
     - Grow in-place: `tree.grow_subtree(id=node_id, X=X_subset, y=y_subset, sample_weight=optional, max_depth=...)`
     - Optional: you may restrict grow_subtree to a subset of features by selecting the corresponding columns as X 
     - Or train in sklearn first and graft: `tree.replace_subtree(node_id, Tree.from_sklearn(trained_subtree))`
     - If the node is not a leaf, prune it first with `tree.prune(node_id)` if you want to replace its children.
   c. Mix methods: fit a subtree in sklearn, prune a bad branch and tweak leaf values manually.
   d. Data/feature manipulations:
     - You may engineer features, drop features, or add sample weights for training.
     - If you edit training entries (X_train or y_train), document the change and rationale. **Do not** modify `y_val` or validation labels.
     - You may remove outliers or add synthetic samples to `X_train` but **never** touch `X_val` or `y_val`.

5. Test targeted local changes:
   - For every change, compute its effect on the validation metric.
   - If a change harms validation, analyze why and either revert or adjust; if it helps, consider whether the improvement is robust and interpretable.
   - Keep a `best_tree` and `best_val_score` as `deepcopy` snapshots to protect against destructive edits.
   - Use clear `print()` labels for all experimental outputs so Observations form a reproducible log.
   - Prefer small, interpretable edits (grafting/pruning, threshold adjustments, local retraining) over end-to-end retraining when possible.

Repeat these steps in any order for as many times as needed, until you no longer have ideas on how to improve the tree.
""".strip()


final_answer_instructions = ("""
Reporting and final answer:
- After you can no longer improve the tree, call `final_answer(...)` inside a `<code>` block. The final answer must include:
  - `summary`: A short paragraph summarizing the final tree and why it was chosen. (Optional: brief discussion of remaining weaknesses and possible next steps.)
  - `train_score`: Final value of the target metric on the training set after all feature/outlier manipulations.
  - `val_score`: Final value of the target metric on the validation set. **Do not remove outliers from `X_val`/`y_val`.**
  - `model`: The final decision tree (an instance of `editable_tree.Tree`).
  - `preprocess_features`: a callable `preprocess_features(X: pd.DataFrame) -> pd.DataFrame` that applies the same preprocessing/feature-engineering used during training.
    - Sanity check: `metric(model.predict(preprocess_features(original_X_val)), y_val) == val_score`
Example final code pattern (for the last message):
<code>
def preprocess_features(X: pd.DataFrame):
    X = deepcopy(X)
    del X["irrelevant_feature"]
    X["new_feature"] = X["feature_A"] + X["feature_B"]
    X = encode_categorical_features(X)
    return X

final_answer({
    "summary": "Short summary text...",
    "train_score": best_train_score, "val_score": best_val_score, "model": best_model,
    "model": best_tree, preprocess_features: preprocess_features, 
})
</code>
""").strip()

metrics_by_task = dict(
    binary="ROC AUC (sklearn.metrics.roc_auc_score)",
    multiclass="Log Loss (sklearn.metrics.log_loss)",
    regression="RMSE (sqrt of sklearn.metrics.mean_squared_error)"
)

starter_snippets_by_task = dict(
    binary="""
<code>
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# you are given input data: X_train/val are pd.DataFrames, y_train/val are 1d np.ndarrays
X_train, X_val, y_train, y_val = preprocess_data_somehow(X_train, X_val, y_train, y_val)  # <-- part of your task

tree = DecisionTreeClassifier(min_samples_split=100)
tree.fit(X_train, y_train)  # important: the tree is trained on a pd.DataFrame
print("Train AUC:", roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]))
print("Val AUC:", roc_auc_score(y_val, tree.predict_proba(X_val)[:, 1]))
from editable_tree import Tree
print(Tree.from_sklearn(tree))
</code>
""".strip(),
    multiclass=f"""
<code>
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

# you are given input data: X_train/val are pd.DataFrames, y_train/val are 1d np.ndarrays
X_train, X_val, y_train, y_val = preprocess_data_somehow(X_train, X_val, y_train, y_val)  # <-- part of your task

tree = DecisionTreeClassifier(min_samples_split=100)
tree.fit(X_train, y_train)  # important: the tree is trained on a pd.DataFrame
print("Train log_loss:", log_loss(y_train, tree.predict_proba(X_train)))
print("Val log_loss:", log_loss(y_val, tree.predict_proba(X_val)))
from editable_tree import Tree
print(Tree.from_sklearn(tree))
</code>
""".strip(),
    regression="""
<code>
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# you are given input data: X_train/val are pd.DataFrames, y_train/val are 1d np.ndarrays
X_train, X_val, y_train, y_val = preprocess_data_somehow(X_train, X_val, y_train, y_val)  # <-- part of your task

tree = DecisionTreeRegressor(criterion="squared_error", min_samples_split=100)
tree.fit(X_train, y_train)  # important: the tree is trained on a pd.DataFrame
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, tree.predict(X_train))))
print("Val RMSE:", np.sqrt(mean_squared_error(y_val, tree.predict(X_val))))
from editable_tree import Tree
print(Tree.from_sklearn(tree))
</code>
""".strip()
)
