import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 在文件开头添加以下代码来解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features_ = int(np.sqrt(self.n_features_))
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)  #
        return 1 - np.sum((counts / m) ** 2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        # 使用当前时间作为随机种子
        np.random.seed(None)  # 使用None或删除这行代码都可以
        features = np.random.choice(self.n_features_, self.max_features_, replace=False)
        for feature in features:
            values = X[:, feature]
            unique_values = np.unique(values)
            for i in range(1, len(unique_values)):
                threshold = (unique_values[i - 1] + unique_values[i]) / 2
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])
                total = len(y)
                weighted_gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / total
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gini

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split or \
                n_labels == 1:
            return {'class': np.argmax(np.bincount(y))}
        best_feature, best_threshold, best_gini = self._best_split(X, y)
        if best_gini == float('inf'):
            return {'class': np.argmax(np.bincount(y))}
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if 'class' in tree:
            return tree['class']
        if x[tree['feature']] < tree['threshold']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

    def feature_importance(self, X, y):
        """计算特征重要性"""
        importances = np.zeros(self.n_features_)
        self._calculate_importance(self.tree, importances)
        return importances / np.sum(importances)

    def _calculate_importance(self, tree, importances):
        """递归计算特征重要性"""
        if 'feature' in tree:
            importances[tree['feature']] += 1
            self._calculate_importance(tree['left'], importances)
            self._calculate_importance(tree['right'], importances)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # 使用当前时间作为随机种子
            np.random.seed(None)  # 使用None或删除这行代码都可以
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    def cross_validate(self, X, y, n_folds=5):
        """添加交叉验证方法"""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True)
        scores = []

        for train_idx, val_idx in kf.split(X):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            self.fit(X_fold_train, y_fold_train)
            y_pred = self.predict(X_fold_val)
            score = np.sum(y_fold_val == y_pred) / len(y_fold_val)
            scores.append(score)

        return np.mean(scores), np.std(scores)

    def feature_importance(self, X, y):
        """计算随机森林的特征重要性"""
        importances = np.zeros(X.shape[1])
        for tree in self.trees:
            importances += tree.feature_importance(X, y)
        return importances / len(self.trees)

    def plot_feature_importance(self, feature_names=None):
        """绘制特征重要性条形图"""
        importances = self.feature_importance(X_train, y_train)
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, importances)
        plt.title('特征重要性分析')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test, class_names=None):
        """绘制混淆矩阵"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        # 修改这里的条件判断
        labels = class_names if isinstance(class_names, (list, type(None))) else 'auto'
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.show()


def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# 主程序
if __name__ == "__main__":
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    # 将类别名称转换为列表
    class_names = list(iris.target_names)

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # 创建并训练随机森林
    rf = RandomForest(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        max_features=2
    )

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测并评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 交叉验证
    cv_mean, cv_std = rf.cross_validate(X_train, y_train)
    print(f"\n交叉验证结果:")
    print(f"平均准确率: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 特征重要性分析
    importances = rf.feature_importance(X_train, y_train)
    print("\n特征重要性:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

    # 可视化
    print("\n正在生成可视化图表...")

    # 1. 特征重要性可视化
    plt.figure(figsize=(12, 6))
    rf.plot_feature_importance(feature_names)

    # 2. 混淆矩阵可视化
    rf.plot_confusion_matrix(X_test, y_test, class_names)

    # 3. 预测结果对比
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, c='blue', label='真实值', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, c='red', label='预测值', alpha=0.5)
    plt.title('预测结果对比')
    plt.xlabel('样本索引')
    plt.ylabel('类别')
    plt.legend()
    plt.tight_layout()
    plt.show()