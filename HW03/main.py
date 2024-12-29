from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
import random
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.decision_tree import entropy, gini
from src.utils import accuracy_score, preprocess, plot_learners_roc


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    feature_names = X_train.columns.tolist()        # Save feature names before preprocessing
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=200,
        learning_rate=0.001,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./adaboost_roc.png',
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    # (TODO) Draw the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance[np.argsort(feature_importance)])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in np.argsort(feature_importance)])
    plt.xlabel('Importance')
    plt.title('Feature Importance - AdaBoost')
    plt.tight_layout()
    plt.savefig('./adaboost_feature_importance.png')
    plt.close()

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=200,
        learning_rate=0.001,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./bagging_roc.png',
    )
    feature_importance = clf_bagging.compute_feature_importance()
    # (TODO) Draw the feature importance
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance[np.argsort(feature_importance)])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in np.argsort(feature_importance)])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Bagging')
    plt.tight_layout()
    plt.savefig('./bagging_feature_importance.png')
    plt.close()

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"Gini index: {gini(y):.4f}")
    print(f"Entropy: {entropy(y):.4f}")

    # Decision Tree - Feature Importance
    feature_importance = clf_tree.compute_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance[np.argsort(feature_importance)])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in np.argsort(feature_importance)])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Decision Tree')
    plt.tight_layout()
    plt.savefig('./decision_tree_feature_importance.png')
    plt.close()


if __name__ == '__main__':
    main()
