"""
A single-file decision tree that supports manual and semi-automatic construction, grafting, pruning and editing.
documentation: see prompting.py
tests: python editable_tree.py
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, List, Dict, Tuple, Union, Sequence
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class Tree:
    """A Tree node that also represents a whole tree using named features.
    Public attributes (for manual editing):
      - id: sklearn-style node id (int)
      - feature: column name (str) or None (None for leaves)
      - threshold: float or None
      - value: np.ndarray (kept flexible; interpreted via `.mean()` for prediction)
      - left, right: Tree or None. The `left` child corresponds to feature<=threshold
    """

    DEPTH_INDENT = "  "
    FEATURE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        /,
        definition: Optional[object] = None,
        *,
        id: Optional[int] = None,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        value: Optional[Union[np.ndarray, int, float]] = None,
        left: Optional[object] = None,
        right: Optional[object] = None,
        le: Optional[object] = None,
        gt: Optional[object] = None,
    ):
        """Construct a Tree node.
        :param id: Optional node identifier. Used for search and editing. Use .repair() to assign automatically.
        :param definition: Optional shorthand initializer, either str (for criterion) or a number (for a leaf value)
            1. str criterion: like "sepal_length <= 0.5" sets feature/threshold.
            2. number/array sets `value` for a leaf node. Equivalent to Tree(value=)
        :param feature: Feature name (string) for split nodes.
        :param threshold: Threshold for split nodes (float).
        :param value: Leaf value (scalar or array).
        :param left: Left child (Tree-like or None) if feature <= threshold. Also available as `le`.
        :param right: Right child (Tree-like or None) if feature > threshold. Also available as `gt`.
        :param le: Alias for `left` aka "less than or equal" child.
        :param gt: Alias for `right` aka "greater than" child.
        :note: If non-Tree children are passed (numbers, arrays, etc), they are wrapped into `Tree(value=child)`
        :raises: ValueError: If `definition` string cannot be parsed, or both alias and canonical child args are
          provided simultaneously, or only one child is provided for a split node.
        """
        self.feature: Optional[str] = None
        self.threshold: Optional[float] = None
        self.value: Optional[np.ndarray] = None
        self.id: Optional[int] = None
        self.left: Optional[Tree] = None
        self.right: Optional[Tree] = None

        # Positional definition parsing
        if definition is not None:
            if isinstance(definition, (int, float, np.integer, np.floating, np.ndarray)):
                value = definition
            elif isinstance(definition, str):
                text_def = definition.strip()
                if "<=" not in text_def:
                    raise ValueError(f"Cannot parse tree definition string: {definition!r}."
                                     " Use syntax: 'feature_name <= threshold'")
                lhs_str, rhs_str = [s.strip() for s in text_def.split("<=", 1)]
                if not self.FEATURE_NAME_RE.match(lhs_str):
                    raise ValueError(f"Cannot parse feature name: {definition!r}."
                                     " Feature names must match /^[A-Za-z_][A-Za-z0-9_]*$/")
                feature = lhs_str
                try:
                    threshold = float(rhs_str)
                except ValueError:
                    raise ValueError(f"Cannot parse threshold: {definition!r}")
            else:
                value = definition

        self.feature, self.threshold = feature, threshold
        # normalize value storage: store as numpy array if provided
        if value is not None and not isinstance(value, np.ndarray):
            # allow scalar numbers too
            try:
                self.value = np.array(value, dtype=float)
            except Exception:
                # fallback: store as object array
                self.value = np.array(value, dtype=object)
        else:
            self.value = value

        self.id = int(id) if id is not None else None

        if le is not None and left is not None:
            raise ValueError("Specify only one of 'le' or 'left'")
        if gt is not None and right is not None:
            raise ValueError("Specify only one of 'gt' or 'right'")
        left = left if left is not None else le
        right = right if right is not None else gt
        if left is not None or right is not None:
            if left is None or right is None:
                raise ValueError("Either both children (left and right) must be provided or neither (leaf).")
            self.left = left if isinstance(left, Tree) else type(self)(left)
            self.right = right if isinstance(right, Tree) else type(self)(right)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outputs for input features using the current tree.
        :param X: pandas.DataFrame with columns that include all feature names used by the tree.
        :returns: numpy array of predictions. If leaf values are scalars, returns shape (n_samples,),
                  if leaf values are vectors of length k, returns shape (n_samples, k).
                  If any node has None value, object dtype with None entries is used (for scalars) or
                  object entries with None or np.ndarray (for vectors).
        :raises: ValueError if X is not a pandas.DataFrame or if inconsistent vector sizes are encountered.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("predict expects a pandas.DataFrame (with named columns).")
        n_samples = X.shape[0]

        # traverse a single sample and return array or None
        def _value_for_sample(i: int) -> Optional[np.ndarray]:
            node = self
            while not node.is_leaf:
                f, thr = node.feature, node.threshold
                if f is None or f not in X.columns:
                    break
                xi = X.iloc[i][f]
                node = node.left if (xi <= thr) else node.right
                if node is None:
                    break
            if node is None or node.value is None:
                return None
            return _value_to_array(node.value).ravel()

        raw_vals: List[Optional[np.ndarray]] = [_value_for_sample(i) for i in range(n_samples)]
        first_non_none = next((v for v in raw_vals if v is not None), None)
        if first_non_none is None:
            out = np.empty((n_samples,), dtype=object)
            out.fill(None)
            return out

        if first_non_none.size == 1:
            if all(v is not None for v in raw_vals):
                out = np.empty((n_samples,), dtype=float)
                for i, v in enumerate(raw_vals):
                    out[i] = float(v[0])
                return out
            out = np.empty((n_samples,), dtype=object)
            for i, v in enumerate(raw_vals):
                out[i] = None if v is None else float(v[0])
            return out
        else:
            k = int(first_non_none.size)
            for v in raw_vals:
                if v is not None and v.size != k:
                    raise ValueError("Inconsistent leaf value vector sizes encountered during prediction.")
            if all(v is not None for v in raw_vals):
                out = np.empty((n_samples, k), dtype=float)
                for i, v in enumerate(raw_vals):
                    out[i, :] = v
                return out
            out = np.empty((n_samples,), dtype=object)
            for i, v in enumerate(raw_vals):
                out[i] = None if v is None else v
            return out

    # -------------------- Prediction and utilities --------------------
    @property
    def is_leaf(self) -> bool:
        """Return True if node has no children."""
        return (self.left is None) and (self.right is None)

    @property
    def le(self) -> "Tree":
        return self.left

    @property
    def gt(self):
        return self.right

    def get_max_depth(self) -> int:
        """Return the maximum depth of this node's subtree (leaf has depth 0)."""
        if self.is_leaf:
            return 0
        left_depth = self.left.get_max_depth() if self.left is not None else -1
        right_depth = self.right.get_max_depth() if self.right is not None else -1
        return 1 + max(left_depth, right_depth)

    def get_num_nodes(self) -> int:
        """Return total number of nodes in this subtree, including self."""
        cnt = 1
        if self.left is not None:
            cnt += self.left.get_num_nodes()
        if self.right is not None:
            cnt += self.right.get_num_nodes()
        return cnt

    @classmethod
    def from_sklearn(cls, tree: Union[DecisionTreeRegressor, DecisionTreeClassifier]) -> "Tree":
        """Build a Tree from a fitted sklearn `DecisionTreeRegressor` or `DecisionTreeClassifier` using named features.
        For classifiers we store class probabilities (not labels). For binary classification we store a single
        scalar: P(class==1). For multiclass we store a probability vector.
        :param tree: A fitted DecisionTreeRegressor or DecisionTreeClassifier instance.
        :returns: Root node of the imported tree.
        :raises: ValueError if the model is not fitted (missing `tree_` attribute) or if feature names are missing.
        """
        if not hasattr(tree, "tree_"):
            raise ValueError("Provided model does not have attribute tree_. Is it fitted?")

        if hasattr(tree, "feature_names_in_"):
            fnames = list(getattr(tree, "feature_names_in_"))
        else:
            raise ValueError("Could not determine feature names for from_sklearn().  Provide `tree.feature_names_in_="
                             "list(...)` or fit the estimator with a DataFrame so that it stores `feature_names_in_`.")

        t = tree.tree_
        n_nodes = t.node_count
        children_left = t.children_left
        children_right = t.children_right
        feature = t.feature
        threshold = t.threshold
        raw_value = t.value

        is_classifier = isinstance(tree, DecisionTreeClassifier)
        nodes: List[Optional[Tree]] = [None] * n_nodes

        for nid in range(n_nodes):
            is_split = feature[nid] >= 0
            feat_name = fnames[int(feature[nid])] if is_split else None
            value_arr = np.array(raw_value[nid]).ravel()
            if is_classifier:
                total = float(value_arr.sum())
                probs = np.zeros_like(value_arr, dtype=float) if total == 0.0 else value_arr.astype(float) / total
                node_value = np.array([probs[1]], dtype=float) if probs.size == 2 else np.array(probs, dtype=float)
            else:
                node_value = np.array(value_arr, dtype=float) if value_arr.size > 0 else None

            nodes[nid] = cls(id=int(nid), feature=feat_name, threshold=float(threshold[nid]) if is_split else None,
                             value=node_value if node_value is not None and node_value.size > 0 else None)

        for nid in range(n_nodes):
            left_id = int(children_left[nid])
            right_id = int(children_right[nid])
            if left_id != -1:
                nodes[nid].left = nodes[left_id]
            if right_id != -1:
                nodes[nid].right = nodes[right_id]

        return nodes[0]

    def _iter_nodes(self) -> List["Tree"]:
        """Return list of all nodes in pre-order."""
        out: List[Tree] = []

        def _recur(n: Optional["Tree"]):
            if n is None:
                return
            out.append(n)
            _recur(n.left)
            _recur(n.right)

        _recur(self)
        return out

    def _max_id(self) -> int:
        """Return the maximum assigned id in the subtree, or -1 if none."""
        indices = [n.id for n in self._iter_nodes() if n.id is not None]
        return max(indices) if indices else -1

    def find_node(self, id: int) -> "Tree":
        """Find a node by its sklearn-style `id`.
        :param id: Node id printed by `print(tree)`.
        :returns: The node with the specified id.
        :raises: ValueError if the node cannot be found.
        """
        self.repair()
        for n in self._iter_nodes():
            if n.id == id:
                return n
        raise ValueError(f"No node with id {id}")

    def get_data_indices_for_node(self, id: int, X: pd.DataFrame) -> np.ndarray:
        """Return indices of rows in X that route to node with given id.

        :param id: Target node id in the current tree.
        :param X: pandas.DataFrame of shape (n_samples, n_features).
        :returns: `np.ndarray` of integer row indices that fall into the specified node
        :raises: ValueError if the node id is not present in the tree.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("get_data_indices_for_node expects a pandas.DataFrame")

        _ = self.find_node(id)  # validates existence
        n_samples = X.shape[0]
        hits: List[np.ndarray] = []

        def _recur(cur: "Tree", mask: np.ndarray):
            if not mask.any():
                return
            if cur.id == id:
                hits.append(np.flatnonzero(mask))
                return
            if cur.is_leaf:
                return
            feat = cur.feature
            thr = cur.threshold
            if feat is None or feat not in X.columns:
                return  # Treat split as not-applicable for this X; cannot descend
            left_mask = mask & (X[feat].values <= thr)
            right_mask = mask & (X[feat].values > thr)
            if cur.left is not None:
                _recur(cur.left, left_mask)
            if cur.right is not None:
                _recur(cur.right, right_mask)

        _recur(self, np.ones(n_samples, dtype=bool))
        if not hits:
            return np.array([], dtype=int)
        return np.concatenate(hits, axis=0)

    @staticmethod
    def _format_numeric(val: Optional[float]) -> str:
        return f"{val:.3f}".rstrip('0').rstrip('.') if val is not None else "None"

    def __repr__(self) -> str:
        """Return a readable text form of the tree using node.id as IDs."""
        try:
            self.repair()
        except Exception:
            pass  # printing should not crash on repair errors

        lines: List[str] = []

        def _format_value(val: Optional[Union[np.ndarray, float, int]]) -> str:
            arr = _value_to_array(val)
            if arr is None:
                return "None"
            if arr.size == 1:
                return self._format_numeric(float(arr.ravel()[0]))
            return f"[{', '.join(Tree._format_numeric(float(x)) for x in arr.ravel())}]"

        def recurse(node: "Tree", depth: int = 0, tag: str = ""):
            indent = self.DEPTH_INDENT * depth
            nid = node.id
            if node.is_leaf:
                lines.append(f"{indent}{tag} leaf value={_format_value(node.value)}  # id: {nid}")
            else:
                l_id = node.left.id if node.left is not None else None
                r_id = node.right.id if node.right is not None else None
                lines.append(f"{indent}{tag} if {node.feature} <= {self._format_numeric(node.threshold)}:  "
                             f"# id: {nid}, then id: {l_id}, else id: {r_id}, value: {_format_value(node.value)}")
                if node.left is not None:
                    recurse(node.left, depth + 1, tag="then:")
                if node.right is not None:
                    recurse(node.right, depth + 1, tag="else:")

        recurse(self)
        return "\n".join(lines)

    def replace_subtree(self, id: int, subtree: "Tree") -> None:
        """Replace the subtree rooted at `id` with a deep copy of `subtree`.
        The target node keeps its original id. All other inserted nodes get fresh,
        unique indices (monotonic, starting after the current max id).
        :param id: id of the node to replace.
        :param subtree: A `Tree` object to graft in place.
        """
        for node in subtree._iter_nodes():
            if node is not subtree:
                node.id = None  # to be repaired later
        target = self.find_node(id)
        target.feature, target.threshold, target.value = subtree.feature, subtree.threshold, subtree.value
        target.left, target.right = subtree.left, subtree.right
        self.repair()

    def prune(self, id: Optional[int] = None, new_value: Optional[np.ndarray] = None) -> None:
        """Turn node with `id` into a leaf.
        :param id: Node id to prune, defaults to self
        :param new_value: Optional new value to assign to that node.
        """
        self.repair()
        node = self.find_node(id) if id is not None else self
        node.left = node.right = node.feature = node.threshold = None
        if new_value is not None:
            node.value = np.array(new_value, dtype=float)

    def grow_subtree(self, *, id: int = None, X: pd.DataFrame, y: np.ndarray, sample_weight=None, tree: str = 'auto', **kwargs) -> None:
        """Extend a leaf node at `id` with a subtree fit via sklearn.

        :param id: id of a node to be extended. If not specified, grows subtree from the current node.
        :param X: Features to train the subtree on (pandas.DataFrame).
        :param y: Targets corresponding to `X`.
        :param sample_weight: Optional array of sample weights.
        :param tree: 'auto', 'classification' or 'regression' - controls which sklearn estimator to use.
        :param kwargs: Extra parameters for `DecisionTreeRegressor` / `DecisionTreeClassifier` (e.g., `max_depth`, `min_samples_split`, etc.).
        :note: The subtree will be fit on the entire provided `X`/`y`. If you want to train on only the samples that
            currently route to this node, first select them via: ids = tree.get_data_indices_for_node(id, X_all)
        :raises: ValueError If the target node is not a leaf or tree type is ambiguous.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("grow_subtree expects a pandas.DataFrame for X")
        if tree not in ('auto', 'classification', 'regression'):
            raise ValueError("tree must be one of 'auto', 'classification', 'regression'")
        y = np.asarray(y)
        tree = tree if tree != 'auto' else infer_tree_type(y)
        estimator = DecisionTreeClassifier(**kwargs) if tree == 'classification' else DecisionTreeRegressor(**kwargs)
        fitted = estimator.fit(X, y, sample_weight=sample_weight)
        subtree = type(self).from_sklearn(fitted)
        self.prune(id)
        self.replace_subtree(id, subtree)

    def fill_values(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None, *,
                    recompute_all: bool = False, fill_empty: bool = None, prune_empty: bool = False) -> None:
        """Fill or recompute node `.value` fields using data routed through the tree.
        This supports both regression targets (scalars) and multiclass targets (one-hot averaged probabilities).
        If the tree already contains values, their shape (scalar vs vector) must be compatible with the provided y,
        otherwise a ValueError is raised.
        :param X: pandas.DataFrame of shape (n_samples, n_features).
        :param y: Target vector of shape (n_samples,).
        :param sample_weight: Optional array of sample weights (same length as `y`).
        :param recompute_all: If False (default), update nodes whose `.value is None`. If True, recompute values for all nodes.
        :param fill_empty: If True, nodes with zero samples/weight copy the nearest ancestor's non-None value.
          Raises `ValueError` if such an ancestor does not exist. If False/None, leaves `.value` as-is for empty nodes.
        :param prune_empty: If True, prune any subtrees with zero samples/weight:
        - If one child has zero routed samples/weight, replace the node with the non-empty child.
        - If both children are empty, prune the node (recursively).
        - The root object is never removed; if replaced by a single leaf, we attach that leaf as `root.left`
            and keep `root` non-leaf to support interactive editing scenarios.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, np.ndarray):
            raise ValueError("fill_values expects a pandas.DataFrame for X and numpy.ndarray for y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must have the same length as y")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative")

        self.repair()

        # Precompute routing masks and weight sums per node
        class_order = None
        node_to_indices: Dict[Tree, np.ndarray] = {}
        node_to_weight: Dict[Tree, float] = {}
        parent_map: Dict[Tree, Optional[Tree]] = {}

        def build_parent_map(node: Optional["Tree"], parent: Optional["Tree"]):
            if node is None:
                return
            parent_map[node] = parent
            build_parent_map(node.left, node)
            build_parent_map(node.right, node)

        build_parent_map(self, None)

        def _assign_masks(node: "Tree", mask: np.ndarray) -> None:
            node_to_indices[node] = np.flatnonzero(mask)
            node_to_weight[node] = float(mask.sum() if sample_weight is None else sample_weight[mask].sum())
            if node.is_leaf or node.feature is None or node.feature not in X.columns:
                return
            left_mask = mask & (X[node.feature].values <= node.threshold)
            right_mask = mask & (X[node.feature].values > node.threshold)
            if node.left is not None:
                _assign_masks(node.left, left_mask)
            if node.right is not None:
                _assign_masks(node.right, right_mask)

        _assign_masks(self, np.ones(len(y), dtype=bool))

        if prune_empty:
            def _prune_rec(node: Optional[Tree]) -> Optional[Tree]:
                if node is None or node.is_leaf:
                    return None if node_to_weight.get(node, 0.0) == 0.0 else node
                left_repl, right_repl = _prune_rec(node.left), _prune_rec(node.right)
                if left_repl is None and right_repl is None:
                    if node_to_weight.get(node, 0.0) > 0.0:  # collapse to leaf
                        node.prune()
                        return node
                    return None
                elif right_repl is None:
                    return right_repl
                elif left_repl is None:
                    return left_repl
                node.left, node.right = left_repl, right_repl
                return node

            repl = _prune_rec(self)
            if repl is None:
                self.prune()
            elif repl is not self:   # become repl, but keep own id
                self.feature, self.threshold, self.value = repl.feature, repl.threshold, repl.value
                self.left, self.right = repl.left, repl.right

            # Recompute masks after pruning
            node_to_indices.clear()
            node_to_weight.clear()
            parent_map.clear()
            build_parent_map(self, None)
            _assign_masks(self, np.ones(len(X), dtype=bool))

        def _mean_for_node(node: Tree) -> Optional[float]:
            idx = node_to_indices.get(node, np.array([], dtype=int))
            if idx.size == 0:
                return None
            if sample_weight is None:
                return np.mean(y[idx], axis=0)
            w = sample_weight[idx]
            wsum = float(w.sum())
            if wsum == 0.0:
                return None
            return np.average(y[idx], weights=w, axis=0)

        existing_val = None
        for n in self._iter_nodes():
            if n.value is not None:
                existing_val = _value_to_array(n.value)
                break

        if existing_val is not None:
            is_multiclass = int(existing_val.size) > 1
        else:
            # infer from y: if non-numeric or integer with multiple unique values -> multiclass
            class_order = np.unique(y)
            if y.dtype.kind in ('O', 'U', 'S'):
                is_multiclass = class_order.size > 1
            elif class_order.size == 2 and set(map(float, class_order)) == {0.0, 1.0}:
                class_order, is_multiclass = np.array([0.0, 1.0]), False
            elif np.issubdtype(y.dtype, np.integer) and class_order.size > 2:
                is_multiclass = True
            else:
                is_multiclass, class_order = False, None  # regression

        for n in self._iter_nodes():
            if n.value is None:
                continue
            arr = _value_to_array(n.value)
            if arr is None:
                continue
            if (arr.size > 1) != is_multiclass:
                raise ValueError("Existing node value formats are incompatible with provided y for fill_values.")

        if is_multiclass:
            class_order = np.asarray(np.unique(y) if class_order is None else class_order)
            y = to_one_hot(y, class_order)

        for node in self._iter_nodes():
            has_samples = node_to_weight.get(node, 0.0) > 0.0
            if not recompute_all and node.value is not None:
                continue
            if has_samples:
                m = _mean_for_node(node)
                if m is not None:
                    node.value = np.array(m, dtype=float)
            if not fill_empty:
                continue
            # Fill from nearest ancestor if requested or mean unavailable
            if (not has_samples and fill_empty) or (has_samples and node.value is None and fill_empty):
                p = parent_map.get(node, None)
                while p is not None:
                    if p.value is not None:
                        node.value = _value_to_array(p.value)
                        break
                    p = parent_map.get(p, None)
                else:
                    raise ValueError(f"Cannot fill value for node id={node.id}: no ancestor with a non-None value.")

        self.repair()  # Keep indices sane after structural changes

    def repair(self, reindex_all: bool = False) -> None:
        """Fix node id issues in-place.
        If `reindex_all=True`, resets all indices and assigns fresh ones, otherwise fill only missing (None) indices.
        For duplicate indices, keeps the node with lowest depth / leftmost order, replacing duplicates with fresh indices.
        :param reindex_all: Whether to reassign indices for all nodes.
        """
        node_meta: List[Tuple[Tree, int, int]] = []  # (node, depth, preorder_idx)

        def _collect(n: Tree, depth: int, counter: List[int]):
            node_meta.append((n, depth, counter[0]))
            counter[0] += 1
            if n.left is not None:
                _collect(n.left, depth + 1, counter)
            if n.right is not None:
                _collect(n.right, depth + 1, counter)

        node_meta.clear()
        _collect(self, 0, [0])

        if reindex_all:
            for n, _, _ in node_meta:
                n.id = None

        id_map: Dict[Optional[int], List[Tuple[Tree, int, int]]] = {}
        for n, depth, order in node_meta:
            id_map.setdefault(n.id, []).append((n, depth, order))

        used_indices = set()
        for idx_key, lst in list(id_map.items()):
            if idx_key is None:
                continue
            if len(lst) == 1:
                used_indices.add(idx_key)
                continue
            lst_sorted = sorted(lst, key=lambda item: (item[1], item[2]))
            used_indices.add(idx_key)
            for other, _, _ in lst_sorted[1:]:
                other.id = None

        max_existing = max(used_indices) if used_indices else -1
        next_free = max_existing + 1
        for n, _, _ in node_meta:
            if n.id is None:
                n.id = next_free
                next_free += 1


def _value_to_array(val: Optional[Union[np.ndarray, float, int]]) -> Optional[np.ndarray]:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    try:
        arr = np.array(val, dtype=float)
        if arr.ndim == 0:
            arr = np.array([float(arr)])
        return arr
    except Exception:
        return np.array(val, dtype=object)


def _is_vector_value(val: Optional[Union[np.ndarray, float, int]]) -> bool:
    arr = _value_to_array(val)
    return False if arr is None else (arr.size > 1)


def to_one_hot(y: np.ndarray, class_order: Sequence) -> np.ndarray:
    mapping = {v: i for i, v in enumerate(class_order)}
    if len(mapping) != len(class_order):
        class_order = np.unique(y)
        mapping = {v: i for i, v in enumerate(class_order)}
    y_onehot = np.zeros((len(y), class_order.size), dtype=float)
    for i_val, v in enumerate(y):
        if v not in mapping:
            raise ValueError("Encountered unknown class while encoding y for fill_values.")
        y_onehot[i_val, mapping[v]] = 1.0
    return y_onehot


def infer_tree_type(y: np.ndarray) -> str:
    uniq = np.unique(y)
    # classification if non-numeric or binary 0/1
    if y.dtype.kind in ('O', 'U', 'S'):
        chosen = 'classification'
    elif uniq.size == 2 and set(map(float, uniq)) == {0.0, 1.0}:
        chosen = 'classification'
    elif np.issubdtype(y.dtype, np.floating):
        # floats that are not simple 0/1 -> regression
        if uniq.size == 2 and set(map(float, uniq)) == {0.0, 1.0}:
            chosen = 'classification'
        else:
            chosen = 'regression'
    elif np.issubdtype(y.dtype, np.integer):
        # integers -> classification
        chosen = 'classification'
    else:
        raise ValueError(
            "Unable to auto-detect task type from y. Please specify tree='classification' or 'regression'")
    if chosen == 'classification' and np.unique(y).size > 10:
        warnings.warn(
            "Growing a classification tree on more than 10 classes; consider using regression or reducing classes.")
    return chosen

# ------------------------- Tests -------------------------


def test_manual_tree():
    print("=== test_manual_tree ===")
    # Construct two equivalent trees using two syntaxes (named features)
    t1 = Tree(feature="f1", threshold=0.5, left=Tree(value=3.5),
              right=Tree(feature="f3", threshold=9, left=Tree(value=-1), right=Tree(value=+1)))
    t2 = Tree("f1 <= 0.5", le=3.5, gt=Tree("f3 <= 9", le=-1, gt=+1))

    # Prepare dummy data as DataFrame
    X = pd.DataFrame([
        [0.1, 0.0, 0.0],   # -> left -> 3.5
        [0.6, 0.0, 5.0],   # -> right then left -> -1
        [0.6, 0.0, 10.0],  # -> right then right -> +1
    ], columns=["f1", "f2", "f3"])
    p1 = t1.predict(X)
    p2 = t2.predict(X)
    assert np.allclose(np.array(p1, dtype=float), np.array([3.5, -1.0, 1.0]))
    assert np.allclose(np.array(p2, dtype=float), np.array([3.5, -1.0, 1.0]))
    print("test_manual_tree PASSED.\n")


def test_init_arguments():
    print("=== test_init_arguments ===")
    Tree("f1 <= 0.5", le=Tree(-3.5), gt=Tree("f3 <= 9", le=-1, gt=+1))
    Tree(3.5)
    Tree(value=3.5)
    Tree(feature="f1", threshold=0.5, left=Tree(1), right=Tree(2))

    try:
        Tree(feature="f1", threshold=1, left=Tree(1), le=Tree(2))
        raise AssertionError("Expected ValueError for both left and le")
    except ValueError:
        pass

    try:
        Tree(feature="f1", threshold=1, right=Tree(1), gt=Tree(2))
        raise AssertionError("Expected ValueError for both right and gt")
    except ValueError:
        pass

    try:
        Tree(feature="f1", threshold=1, left=Tree(1))
        raise AssertionError("Expected ValueError for single child")
    except ValueError:
        pass

    # invalid: definition string must be 'name <= number'
    try:
        Tree("X[:, 0] <= 0.5", le=Tree(1), gt=Tree(2))
        raise AssertionError("Expected ValueError for X[:, ...] style in definition string")
    except ValueError:
        pass

    print("test_init_arguments PASSED.\n")


def test_grow_subtree():
    print("=== test_grow_subtree ===")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X_raw, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X_raw, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    tree = Tree(id=0)
    tree.grow_subtree(id=0, X=X_train, y=y_train, max_depth=2)
    assert not tree.is_leaf
    # try classification grow as well
    tree2 = Tree(id=100)
    tree2.grow_subtree(id=100, X=X_train, y=y_train, tree='classification', max_depth=2)
    assert not tree2.is_leaf

    id_child = tree.right.right.id
    assert tree2.right.right.is_leaf
    ids = tree.get_data_indices_for_node(id_child, X_train)
    tree.grow_subtree(id=id_child, X=X_train.iloc[ids], y=y_train[ids], max_depth=1)
    assert not tree.right.right.is_leaf
    assert tree.right.right.left.is_leaf and tree.right.right.right.is_leaf
    print("test_grow_subtree PASSED.\n")


def test_fill_values():
    print("=== test_fill_values ===")
    X3 = pd.DataFrame([[0.6], [0.7], [0.8]], columns=["f1"])
    y3 = np.array([10.0, 10.0, 10.0])
    t3 = Tree("f1 <= 0.5", le=3.0, gt=Tree("f2 <= 0.1", le=4.0, gt=5.0))
    assert t3.value is None
    t3.fill_values(X3, y3, recompute_all=True, fill_empty=True)
    assert not t3.is_leaf and t3.value is not None
    t3.fill_values(X3, y3, prune_empty=True)
    assert t3.is_leaf  # tree became a leaf
    preds = np.array(t3.predict(X3), dtype=float)
    assert np.allclose(preds, np.full_like(preds, 10.0))
    w = np.array([1.0, 0.0, 0.0])
    t3.fill_values(X3, y3, sample_weight=w, recompute_all=True, fill_empty=True)
    preds_w = np.array(t3.predict(X3), dtype=float)
    assert np.allclose(preds_w, np.full_like(preds_w, 10.0))
    print("test_fill_values PASSED.\n")


def test_predict_all_none():
    print("=== test_predict_all_none ===")
    root = Tree(feature="x", threshold=0.5, left=Tree(value=None), right=Tree(value=None))
    X = pd.DataFrame([[0.1], [0.6]], columns=["x"])
    preds = root.predict(X)
    assert preds.dtype == object
    assert preds.shape == (2,)
    assert preds[0] is None and preds[1] is None
    print("test_predict_all_none PASSED.\n")


def test_predict_inconsistent_vectors():
    print("=== test_predict_inconsistent_vectors ===")
    root = Tree(feature="x", threshold=0.5, left=np.array([1.0, 2.0]), right=np.array([1.0, 2.0, 3.0]))
    X = pd.DataFrame([[0.1], [0.6]], columns=["x"])
    try:
        _ = root.predict(X)
        raise AssertionError("Expected ValueError due to inconsistent vector sizes")
    except ValueError:
        pass
    print("test_predict_inconsistent_vectors PASSED.\n")


def test_editing(regression: bool = True):
    print("=== test_editing (regression=%s) ===" % regression)
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    X_raw, y = load_iris(return_X_y=True)
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = pd.DataFrame(X_raw, columns=cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    if regression:
        model = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
        original_val_mse = mean_squared_error(y_val, model.predict(X_val))
    else:
        model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        classes = model.classes_
        y_val = to_one_hot(y_val, classes).reshape((len(y_val), -1))
        original_val_mse = mean_squared_error(y_val, proba)

    print("Val MSE with original subtree:", original_val_mse)
    tree = Tree.from_sklearn(model)
    print(f"Tree depth: {tree.get_max_depth()}, size: {tree.get_num_nodes()}")
    print(tree, "\n")
    print(f"Root node: {tree.id=}, {tree.feature=} (name), {tree.threshold=}\n")
    print(f"Left subtree, where {tree.feature} <= {tree.threshold}:\n{tree.left}\n")
    print(f"Right subtree, where {tree.feature} > {tree.threshold}:\n{tree.right}\n")

    node = next((n for n in tree._iter_nodes() if not n.is_leaf), tree)
    print(f"Node {node.id} subtree depth: {node.get_max_depth()}, subtree size: {node.get_num_nodes()}")
    print(f"Node {node.id} subtree structure:\n", node, "\n")

    ids_train = tree.get_data_indices_for_node(node.id, X_train)
    ids_val = tree.get_data_indices_for_node(node.id, X_val)
    print(f"Num samples at node {node.id}: train={len(ids_train)}, val={len(ids_val)}")

    if regression:
        subtree_model = DecisionTreeRegressor(max_depth=2).fit(X_train.iloc[ids_train], y_train[ids_train])
        y_val_pred_for_subtree = subtree_model.predict(X_val.iloc[ids_val])
    else:
        subtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train.iloc[ids_train], y_train[ids_train])
        proba_sub = subtree_model.predict_proba(X_val.iloc[ids_val])
        y_val_pred_for_subtree = proba_sub

    print("Val partial MSE for subtree:", np.nan if y_val_pred_for_subtree is None else 0.0)
    modified_tree = deepcopy(tree)
    modified_tree.replace_subtree(node.id, Tree.from_sklearn(subtree_model))
    modified_val_mse = mean_squared_error(y_val, modified_tree.predict(X_val))
    print(f"Val MSE with grafted subtree: {modified_val_mse}, better? =", modified_val_mse < original_val_mse)

    prune_candidate = next((n for n in modified_tree._iter_nodes() if n.get_max_depth() > 0 and n is not modified_tree), None)
    assert prune_candidate is not None
    pid = prune_candidate.id
    assert modified_tree.find_node(id=pid).get_max_depth() != 0
    modified_tree.prune(id=pid)
    assert modified_tree.find_node(id=pid).get_max_depth() == 0
    y_pred = modified_tree.predict(X_val)
    modified_tree.repair(reindex_all=True)
    modified_tree.fill_values(X_train, y_train, recompute_all=True, prune_empty=True)
    y_pred_after_reindex = modified_tree.predict(X_val)
    assert np.allclose(y_pred, y_pred_after_reindex)
    print(f"test_editing({regression=}) passed\n")


if __name__ == "__main__":
    try:
        test_manual_tree()
        test_init_arguments()
        test_grow_subtree()
        test_fill_values()
        test_predict_all_none()
        test_predict_inconsistent_vectors()
        test_editing(regression=True)
        test_editing(regression=False)
        print("All tests passed.")
    except Exception as e:
        print("A test failed with exception:")
        raise
