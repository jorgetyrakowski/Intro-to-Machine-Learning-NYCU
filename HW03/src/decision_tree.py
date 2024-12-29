"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.n_features = None

    def fit(self, X, y):
        """Train the decision tree"""
        # Store number of features
        _, self.n_features = X.shape
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1:
            return {'value': np.bincount(y).argmax()}
            
        # Find best split
        best_feature, best_threshold = find_best_split(X, y)
        
        if best_feature is None:
            return {'value': np.bincount(y).argmax()}
            
        # Split dataset
        left_mask, right_mask = split_dataset(X, y, best_feature, best_threshold)
        
        # Create child nodes
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'value': np.bincount(y).argmax()
        }

    def predict(self, X):
        """Predict class for X"""
        return np.array([self._predict_tree(x, self.tree) for x in X])
        
    def _predict_tree(self, x, tree_node):
        """Traverse tree for prediction"""
        if 'feature' not in tree_node:
            return tree_node['value']
            
        if x[tree_node['feature']] <= tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        return self._predict_tree(x, tree_node['right'])
    
    def compute_feature_importance(self):
        """
        Compute feature importance based on information gain at each split
        
        Returns:
            np.array: Feature importance scores normalized to sum to 1
        """
        feature_importance = np.zeros(self.n_features)
        self._compute_importance(self.tree, feature_importance, 1.0)
        
        # Normalize
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance

    def _compute_importance(self, node, importance_array, weight):
        """
        Recursively compute feature importance
        
        Args:
            node: Current tree node
            importance_array: Array to store importance values
            weight: Current node weight (decreases with depth)
        """
        if 'feature' not in node:
            return
            
        # Add importance to this feature
        importance_array[node['feature']] += weight
        
        # Recurse on children with reduced weight
        if 'left' in node:
            self._compute_importance(node['left'], importance_array, weight * 0.5)
        if 'right' in node:
            self._compute_importance(node['right'], importance_array, weight * 0.5)

# Functions outside the class remain the same
def split_dataset(X, y, feature_index, threshold):
    """Split dataset based on a feature and threshold"""
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return left_mask, right_mask

def find_best_split(X, y):
    """Find the best split for the dataset"""
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    n_samples, n_features = X.shape
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        
        for threshold in thresholds:
            left_mask, right_mask = split_dataset(X, y, feature, threshold)
            
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                continue
                
            gain = entropy(y) - (
                len(y[left_mask]) / n_samples * entropy(y[left_mask]) +
                len(y[right_mask]) / n_samples * entropy(y[right_mask])
            )
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold

def entropy(y):
    """Calculate entropy of an array"""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def gini(y):
    """Calculate Gini index of an array"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)