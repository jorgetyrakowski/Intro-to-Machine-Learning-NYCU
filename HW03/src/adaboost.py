import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).reshape(-1, 1)
            
        n_samples = len(X_train)
        self.sample_weights = torch.ones(n_samples) / n_samples
        
        losses_of_models = []
        
        for model in self.learners:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model_losses = []
            
            # Train current weak learner
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                outputs = model(X_train)
                
                # Weighted BCE loss
                loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, y_train)
                weighted_loss = (loss.squeeze() * self.sample_weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                model_losses.append(weighted_loss.item())
            
            # Calculate error and alpha
            with torch.no_grad():
                predictions = torch.sigmoid(model(X_train))
                incorrect = (predictions.round() != y_train).squeeze()
                error = (incorrect * self.sample_weights).sum()
                error = torch.clamp(error, 1e-10, 1 - 1e-10)
                
                alpha = 0.5 * torch.log((1 - error) / error)
                self.alphas.append(alpha.item())
                
                # Update weights
                self.sample_weights *= torch.exp(alpha * incorrect)
                self.sample_weights /= self.sample_weights.sum()
            
            losses_of_models.append(model_losses)
        
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        # Get individual predictions
        individual_predictions = []
        ensemble_prediction = np.zeros(len(X))
        
        with torch.no_grad():
            for i, (model, alpha) in enumerate(zip(self.learners, self.alphas)):
                probs = torch.sigmoid(model(X)).numpy().squeeze()
                individual_predictions.append(probs)
                ensemble_prediction += alpha * (2 * probs - 1)
        
        # Final class prediction
        final_predictions = (ensemble_prediction > 0).astype(int)
        
        return final_predictions, individual_predictions
    

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        Compute feature importance for AdaBoost with two-layer model
        
        For a two-layer model, we compute importance by considering how each input
        feature contributes to the final output through both layers.
        
        Returns:
            t.Sequence[float]: Feature importance scores normalized to sum to 1
        """
        importance_scores = np.zeros(self.learners[0].layer1.weight.shape[1])
        total_alpha = sum(abs(alpha) for alpha in self.alphas)
        
        for model, alpha in zip(self.learners, self.alphas):
            # Get weights from both layers
            W1 = model.layer1.weight.data.numpy()  # (hidden_dim, input_dim)
            W2 = model.layer2.weight.data.numpy()  # (1, hidden_dim)
            
            # Compute contribution of each feature through both layers
            # W2: (1, hidden_dim), W1: (hidden_dim, input_dim)
            W2 = W2.reshape(-1)  # Convert to (hidden_dim,)
            combined_weights = np.zeros(W1.shape[1])  # (input_dim,)
            
            # Compute importance for each feature
            for i in range(W1.shape[1]):  # For each input feature
                feature_contrib = np.abs(W2 @ W1[:, i])  # Contribution through hidden layer
                combined_weights[i] = feature_contrib
            
            # Weight by alpha and add to total importance
            importance_scores += combined_weights * abs(alpha) / total_alpha
        
        # Normalize to sum to 1
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return importance_scores