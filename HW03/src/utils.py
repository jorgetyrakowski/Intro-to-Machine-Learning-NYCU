import typing as t
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    Preprocess the input DataFrame for machine learning models.
    
    Steps:
    1. Handle categorical variables using Label Encoding
    2. Scale numerical variables using StandardScaler
    3. Convert to numpy array
    
    Args:
        df (pd.DataFrame): Input DataFrame with raw features
        
    Returns:
        np.ndarray: Preprocessed features ready for model training
    """
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership',
                       'loan_intent', 'previous_loan_defaults_on_file']
    numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'target']
    
    # Initialize encoders dictionary
    encoders = {}
    
    # Handle categorical variables
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df_processed[col] = encoders[col].fit_transform(df_processed[col])
    
    # Scale numerical variables
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # Convert to numpy array
    return df_processed.to_numpy()

class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        # Two layer implementation
        hidden_dim = 16
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        """
        Forward pass of the classifier.
        No non-linear activation is used as per requirements.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Linear output of shape (batch_size, 1)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Linear transformations without any activation function
        x = self.layer1(x)  # First linear layer
        x = self.layer2(x)  # Second linear layer
        return x


def accuracy_score(y_trues, y_preds) -> float:
    """
    Calculate the accuracy score.
    
    Args:
        y_trues (np.ndarray): True labels
        y_preds (np.ndarray): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    return np.mean(y_trues == y_preds)


def entropy_loss(outputs, targets):
    """
    Calculate the binary cross entropy loss.
    
    Args:
        outputs (torch.Tensor): Model outputs (logits)
        targets (torch.Tensor): True labels
        
    Returns:
        torch.Tensor: Computed loss
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(outputs)
    
    # Compute binary cross entropy manually
    epsilon = 1e-15  # Small constant to avoid log(0)
    probs = torch.clamp(probs, epsilon, 1 - epsilon)
    loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    
    return loss.mean()


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    """
    Plot ROC curves for each weak learner.
    
    Args:
        y_preds (List[Sequence[float]]): List of prediction probabilities from each learner
        y_trues (Sequence[int]): True labels
        fpath (str): File path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each learner
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {i+1} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Weak Learners')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(fpath)
    plt.close()

def plot_feature_importance(feature_names: t.List[str], 
                          importance_scores: np.ndarray,
                          title: str,
                          fpath: str):
    """
    Plot feature importance horizontally.
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        title: Plot title
        fpath: File path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bars
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importance_scores[indices])
    
    # Set feature names as y-axis labels
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()