import numpy as np
import pandas as pd

class SimpleDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        self.tree = self._build_tree(data, depth=0)
        self.most_common_class = y.mode()[0]

    def predict(self, X):
        return X.apply(self._predict_row, axis=1, args=(self.tree,))

    def _entropy(self, y):
        proportions = y.value_counts(normalize=True)
        return -sum(proportions * np.log2(proportions))

    def _information_gain(self, data, feature, target_name):
        total_entropy = self._entropy(data[target_name])
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = sum((counts[i] / sum(counts)) * self._entropy(data[data[feature] == values[i]][target_name]) for i in range(len(values)))
        return total_entropy - weighted_entropy

    def _best_split(self, data, target_name):
        features = data.columns.drop(target_name)
        best_feature = max(features, key=lambda feature: self._information_gain(data, feature, target_name))
        return best_feature

    def _build_tree(self, data, depth):
        target_name = 'target'
        if len(data[target_name].unique()) == 1:
            return data[target_name].iloc[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return data[target_name].mode()[0]
        best_feature = self._best_split(data, target_name)
        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            subtree = self._build_tree(data[data[best_feature] == value], depth + 1)
            tree[best_feature][value] = subtree
        return tree

    def _predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = row[feature]
        subtree = tree[feature].get(value, None)
        if subtree is None:
            return self.most_common_class
        return self._predict_row(row, subtree)

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SimpleDecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")