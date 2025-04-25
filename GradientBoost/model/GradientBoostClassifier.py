import numpy as np


class DecisionTreeRegressor:
    """Simple CART regression tree for squared-error splits."""
    class Node:
        def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build(X, y, depth=0)
    
    def _build(self, X, y, depth):
        # stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.std(y) < 1e-8:
            leaf_val = np.mean(y)
            return DecisionTreeRegressor.Node(value=leaf_val)
        
        # find best split
        best_feat, best_thr, best_loss = None, None, np.inf
        for j in range(self.n_features):
            xs = X[:, j]
            thresholds = np.unique(xs)
            for t in (thresholds[:-1] + thresholds[1:]) / 2:
                left_idx = xs <= t
                right_idx = xs > t
                if left_idx.sum() < 1 or right_idx.sum() < 1:
                    continue
                left_y, right_y = y[left_idx], y[right_idx]
                loss = ((left_y - left_y.mean())**2).sum() + ((right_y - right_y.mean())**2).sum()
                if loss < best_loss:
                    best_feat, best_thr, best_loss = j, t, loss
        
        if best_feat is None:
            return DecisionTreeRegressor.Node(value=np.mean(y))
        
        left_idx = X[:, best_feat] <= best_thr
        right_idx = ~left_idx
        left = self._build(X[left_idx], y[left_idx], depth+1)
        right = self._build(X[right_idx], y[right_idx], depth+1)
        return DecisionTreeRegressor.Node(feature=best_feat, threshold=best_thr, left=left, right=right)
    
    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])
    
    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)


class GradientBoostClassifier:
    """
    Binary classifier via gradient boosting on logistic loss.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.F0 = None

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        y = y.astype(float)
        # initialize raw prediction F0 = log(p / (1-p))
        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.F0 = np.log(p / (1 - p))
        Fm = np.full(y.shape, self.F0)

        for _ in range(self.n_estimators):
            # negative gradient = y - p
            p_m = self._sigmoid(Fm)
            residual = y - p_m

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            Fm += self.learning_rate * update

            self.trees.append(tree)

    def predict_proba(self, X):
        Fm = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        p = self._sigmoid(Fm)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
