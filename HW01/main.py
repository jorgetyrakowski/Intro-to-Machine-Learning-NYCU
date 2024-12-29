import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # Add a column of ones to X (for the intercept term)
        # X.shape[0] gives the number of rows (data points)
        ones = np.ones((X.shape[0], 1))

        # Combine the ones (intercept) with the input features X
        # np.hstack stacks arrays horizontally (adds the ones column to the left of X)
        X_b = np.hstack((ones, X))

        # Compute the first part of the normal equation:
        # np.linalg.inv calculates the inverse of a matrix
        # X_b.T is the transpose of X_b, and X_b.T.dot(X_b) is the dot product
        theta_part1 = np.linalg.inv(X_b.T.dot(X_b))

        # Compute the second part of the normal equation:
        # X_b.T.dot(y) is the dot product of the transpose of X_b and the target values y
        theta_part2 = X_b.T.dot(y)

        # Calculate theta (parameters of the linear regression) by multiplying both parts
        theta = theta_part1.dot(theta_part2)

        # Set the intercept (theta[0] is the bias term or intercept)
        self.intercept = theta[0]

        # Set the weights (theta[1:] contains the coefficients for each feature in X)
        self.weights = theta[1:]

    def predict(self, X):
        # Add a column of ones to X for the intercept term (same as during training)
        ones = np.ones((X.shape[0], 1))

        # Combine the ones column with the input features X
        X_b = np.hstack((ones, X))

        # Concatenate the intercept and the weights to create a single vector of parameters
        # Perform the dot product to calculate the predictions for each input in X
        return X_b.dot(np.concatenate(([self.intercept], self.weights)))


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        # Get the number of samples (rows) and features (columns) from X
        num_samples, num_features = X.shape

        # Ensure y is a 1D array by flattening it if it has more than one dimension
        if len(y.shape) > 1:
            y = y.flatten()

        # Initialize the weights (one for each feature) to zeros
        # The intercept (bias term) is also initialized to 0
        self.weights = np.zeros(num_features)
        self.intercept = 0

        # Create a list to store the loss (error) values for each epoch
        losses = []

        # Training loop: repeat the process for the specified number of epochs
        for epoch in range(epochs):
            # 1 Make predictions by calculating the dot product of X and weights, then adding the intercept
            y_pred = np.dot(X, self.weights) + self.intercept

            # 2 Calculate the error (difference between predictions and actual values)
            error = y_pred - y

            # 3 Calculate the gradients for the weights (dw) and intercept (db)
            # dw tells us how much to adjust each weight to reduce the error
            dw = (1 / num_samples) * np.dot(X.T, error)
            # db tells us how much to adjust the intercept
            db = (1 / num_samples) * np.sum(error)

            # 4 Update the weights and intercept by moving them in the direction of the negative gradient
            # The learning_rate controls the size of each step
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db

            # 5 Calculate the Mean Squared Error (MSE) loss for the current predictions
            mse_loss = np.mean(error ** 2)

            # Log the loss every 10,000 epochs to track progress
            if epoch % 10000 == 0:
                logger.info(f"Epoch {epoch}, Loss: {mse_loss:.4f}")

            # Save the loss for this epoch
            losses.append(mse_loss)

        # Return the list of loss values to see how the model improved over time
        return losses

    def predict(self, X):
        # To make predictions, calculate the dot product of X and the learned weights, then add the learned intercept
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        # Create a plot of the losses over the epochs
        plt.plot(range(len(losses)), losses, label='Training Loss')
        # Label the x-axis as 'Epochs' (the number of iterations)
        plt.xlabel('Epochs')
        # Label the y-axis as 'Mean Squared Error' (the loss function)
        plt.ylabel('Mean Squared Error')
        # Title of the plot
        plt.title('Learning Curve')
        # Add a legend to the plot
        plt.legend()
        # Add a grid for better readability
        plt.grid(True)
        # Display the plot
        plt.show()


def compute_mse(prediction, ground_truth):
    # The MSE is the average of the squared differences between the predicted and actual values
    return np.mean((prediction - ground_truth) ** 2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.000382, epochs=405000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
