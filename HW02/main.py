import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        # Get number of samples and features
        n_samples, n_features = inputs.shape

        # Initialize weights and intercept
        # We use small random values for weights to break symmetry
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.randn(n_features) * 0.01
        self.intercept = 0

        # Gradient Descent
        for iteration in range(self.num_iterations):
            # Forward pass
            # 1: Calculate linear combination (z = X.w + b)
            z = np.dot(inputs, self.weights) + self.intercept
            # 2: Apply sigmoid to get predictions
            y_predicted = self.sigmoid(z)

            # Calculate gradients
            # For Cross-Entropy loss, the gradients simplify to:
            # dw = (1/m) * X^T · (predictions - y)
            # db = (1/m) * sum(predictions - y)
            dw = (1 / n_samples) * np.dot(inputs.T, (y_predicted - targets))
            db = (1 / n_samples) * np.sum(y_predicted - targets)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        # Calculate the linear combination
        z = np.dot(inputs, self.weights) + self.intercept
        # Get probabilities using sigmoid
        probabilities = self.sigmoid(z)
        # Get class predictions (0 or 1) using threshold of 0.5
        predictions = (probabilities >= 0.5).astype(int)

        return probabilities, predictions

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        # We use np.exp() for calculating e^(-x)
        # and np.clip to prevent overflow in exp
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # Separate data by class
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        # Calculate mean vectors for each class
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # Calculate scatter matrices
        # Within-class scatter matrix Sw = S0 + S1
        S0 = np.zeros((2, 2))
        for x in X0:
            x = x.reshape(-1, 1)  # Make column vector
            m0 = self.m0.reshape(-1, 1)
            S0 += (x - m0) @ (x - m0).T

        S1 = np.zeros((2, 2))
        for x in X1:
            x = x.reshape(-1, 1)  # Make column vector
            m1 = self.m1.reshape(-1, 1)
            S1 += (x - m1) @ (x - m1).T

        self.sw = S0 + S1

        # Between-class scatter matrix Sb
        m0 = self.m0.reshape(-1, 1)
        m1 = self.m1.reshape(-1, 1)
        self.sb = (m1 - m0) @ (m1 - m0).T

        # Calculate Fisher's linear discriminant w = Sw^(-1)(m1 - m0)
        try:
            self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
            # Normalize w
            self.w = self.w / np.linalg.norm(self.w)

            # Calculate slope for plotting
            self.slope = -self.w[0] / self.w[1]

        except np.linalg.LinAlgError:
            # Handle singular matrix
            logger.error("Singular matrix encountered. Using alternative calculation.")
            self.w = self.m1 - self.m0
            self.w = self.w / np.linalg.norm(self.w)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        # Project data
        projections = inputs @ self.w

        # Project means
        m0_proj = self.m0 @ self.w
        m1_proj = self.m1 @ self.w

        # Calculate distances to projected means
        dist_to_m0 = np.abs(projections - m0_proj)
        dist_to_m1 = np.abs(projections - m1_proj)

        # Classify based on nearest projected mean
        return (dist_to_m1 < dist_to_m0).astype(int)

    def plot_projection(self, inputs: npt.NDArray[float], train_inputs=None, train_targets=None):
        """
        Plot FLD projection with correct perpendicular projections to decision boundary
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get predictions
        predictions = self.predict(inputs)
        
        # Calculate decision boundary
        midpoint = (self.m0 + self.m1) / 2
        slope = -self.w[0] / self.w[1]
        intercept = midpoint[1] - slope * midpoint[0]
        
        # Plot decision boundary
        x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
        x_line = np.array([x_min, x_max])
        y_line = slope * x_line + intercept
        
        ax1.plot(x_line, y_line, 'k-', linewidth=2, label='Decision boundary')
        
        # Calculate direction vectors
        # Vector parallel to decision boundary
        v_parallel = np.array([1, slope])
        v_parallel = v_parallel / np.linalg.norm(v_parallel)
        # Vector perpendicular to decision boundary
        v_perp = np.array([-slope, 1])
        v_perp = v_perp / np.linalg.norm(v_perp)
        
        projected_points = []
        for i, point in enumerate(inputs):
            # Calculate projection point on decision boundary
            # Using point-to-line projection formula
            # p' = p + ((a·x₀ + b·y₀ + c)/(a² + b²))·(-a,b)
            # where ax + by + c = 0 is the line equation
            a = slope
            b = -1
            c = intercept
            
            num = a * point[0] + b * point[1] + c
            denom = a * a + b * b
            
            proj_x = point[0] - num * a / denom
            proj_y = point[1] - num * b / denom
            
            proj_point = np.array([proj_x, proj_y])
            projected_points.append(num/np.sqrt(denom))
            
            # Plot projection line
            color = 'blue' if predictions[i] == 0 else 'red'
            ax1.plot([point[0], proj_x], [point[1], proj_y], 
                    color=color, alpha=0.3, zorder=1)
        
        # Plot test points
        ax1.scatter(inputs[predictions == 0, 0], inputs[predictions == 0, 1],
                    c='blue', s=50, label='Test Class 0', zorder=2)
        ax1.scatter(inputs[predictions == 1, 0], inputs[predictions == 1, 1],
                    c='red', s=50, label='Test Class 1', zorder=2)
        
        # Plot mean points
        ax1.scatter(self.m0[0], self.m0[1], c='blue', marker='*', s=200, 
                    label='Mean Class 0', zorder=3)
        ax1.scatter(self.m1[0], self.m1[1], c='red', marker='*', s=200, 
                    label='Mean Class 1', zorder=3)
        
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('Feature 10')
        ax1.set_ylabel('Feature 20')
        ax1.set_title(f'FLD Projection (slope={slope:.4f}, intercept={intercept:.4f})')
        ax1.legend()
        
        # 1D projection plot
        projected_points = np.array(projected_points)
        
        # Project means onto perpendicular direction
        m0_proj = (a * self.m0[0] + b * self.m0[1] + c) / np.sqrt(a*a + b*b)
        m1_proj = (a * self.m1[0] + b * self.m1[1] + c) / np.sqrt(a*a + b*b)
        
        # Plot 1D projections
        ax2.scatter(projected_points[predictions == 0], 
                    np.zeros_like(projected_points[predictions == 0]),
                    c='blue', label='Class 0')
        ax2.scatter(projected_points[predictions == 1], 
                    np.zeros_like(projected_points[predictions == 1]),
                    c='red', label='Class 1')
        
        # Plot projected means
        ax2.axvline(x=m0_proj, color='blue', linestyle='--', label='Mean Class 0')
        ax2.axvline(x=m1_proj, color='red', linestyle='--', label='Mean Class 1')
        
        ax2.grid(True)
        ax2.set_xlabel('Projected Values')
        ax2.set_title('1D Projection Space')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def compute_auc(y_trues, y_preds):
    """
    Compute Area Under the Curve (AUC) score.

    Parameters:
    y_trues: array-like of shape (n_samples,)
        Ground truth (correct) labels
    y_preds: array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    Returns:
    float: AUC score
    """
    # We can use sklearn's roc_auc_score as mentioned in the slides
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    """
    Compute accuracy score.

    Parameters:
    y_trues: array-like of shape (n_samples,)
        Ground truth (correct) labels
    y_preds: array-like of shape (n_samples,)
        Predicted labels

    Returns:
    float: The fraction of correctly classified samples
    """
    # Convert inputs to numpy arrays for easier computation
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    # Calculate accuracy: (number of correct predictions) / (total number of predictions)
    return np.mean(y_trues == y_preds)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.008,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    # Fit the FLD model
    FLD_.fit(x_train, y_train)
    # Make predictions
    y_pred = FLD_.predict(x_test)
    # Compute the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    # Plot the results
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
