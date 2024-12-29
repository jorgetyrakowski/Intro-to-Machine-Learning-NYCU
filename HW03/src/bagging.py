import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        
        n_samples = len(X_train)
        losses_of_models = []
        
        for model in self.learners:
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]
            
            # Setup optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model_losses = []
            
            # Train on bootstrap sample
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                outputs = model(X_bootstrap)
                loss = nn.BCEWithLogitsLoss()(outputs, y_bootstrap)
                
                loss.backward()
                optimizer.step()
                
                model_losses.append(loss.item())
                
            losses_of_models.append(model_losses)
            
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        # Get individual predictions
        individual_predictions = []
        with torch.no_grad():
            for model in self.learners:
                probs = torch.sigmoid(model(X)).numpy().squeeze()
                individual_predictions.append(probs)
        
        # Average predictions
        ensemble_predictions = np.mean(individual_predictions, axis=0)
        
        # Return both class predictions and individual probabilities
        return (ensemble_predictions > 0.5).astype(int), individual_predictions


    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        Compute feature importance for Bagging with two-layer model
        
        For a two-layer model, we compute importance by considering how each input
        feature contributes to the final output through both layers.
        
        Returns:
            t.Sequence[float]: Feature importance scores normalized to sum to 1
        """
        importance_scores = np.zeros(self.learners[0].layer1.weight.shape[1])
        
        for model in self.learners:
            # Get weights from both layers
            W1 = model.layer1.weight.data.numpy()  # (hidden_dim, input_dim)
            W2 = model.layer2.weight.data.numpy()  # (1, hidden_dim)
            
            # Compute contribution of each feature through both layers
            W2 = W2.reshape(-1)  # Convert to (hidden_dim,)
            combined_weights = np.zeros(W1.shape[1])  # (input_dim,)
            
            # Compute importance for each feature
            for i in range(W1.shape[1]):  # For each input feature
                feature_contrib = np.abs(W2 @ W1[:, i])  # Contribution through hidden layer
                combined_weights[i] = feature_contrib
                
            importance_scores += combined_weights
        
        # Average across models and normalize
        importance_scores = importance_scores / len(self.learners)
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return importance_scores