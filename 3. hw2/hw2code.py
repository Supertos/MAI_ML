import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    idx = np.argsort( feature_vector )
    f, t = feature_vector[idx], target_vector[idx]
    thrs = (f[:-1] + f[1:]) / 2
    mask = f[1:] != f[:-1]
    thrs, n = thrs[mask], len(t)
    if not len(thrs):
        return np.array([]), np.array([]), None, -np.inf
    
    nl = np.arange(1, n)[mask]
    nr = n - nl
    cl = np.cumsum(t)[:-1][mask]
    cr = np.sum(t) - cl
    
    p1l, p1r = cl / nl, cr / nr
    Hl = 1 - p1l**2 - (1 - p1l)**2
    Hr = 1 - p1r**2 - (1 - p1r)**2
    ginis = -(nl / n) * Hl - (nr / n) * Hr
    
    i = np.argmax( ginis )
    return thrs, ginis, thrs[i], ginis[i]


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        super().__init__()
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratios = {}
                for cat in counts:
                    n = counts[cat]
                    c = clicks.get(cat, 0)
                    ratios[cat] = c / n if n > 0 else 0.0
                sorted_categories = sorted(ratios, key=ratios.get)
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"] = {}
        node["right_child"] = {}
        left_mask = split
        right_mask = np.logical_not(split)
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"])
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature = node["feature_split"]
        if self.feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self.feature_types,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self