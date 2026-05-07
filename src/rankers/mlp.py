from typing import Optional, Tuple
import numpy as np
from .base import BaseRanker
import torch
import torch.nn as nn


class MLPRanker(BaseRanker):
    """
    MLP-based ranker with a listwise ListNet loss.
    For each group (target language), the predicted scores are softmaxed across
    its sources and compared to the softmaxed relevance distribution via cross-entropy.
    """

    def __init__(self,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 learning_rate: float = 1e-3,
                 epochs: int = 200,
                 dropout: float = 0.1,
                 weight_decay: float = 0.0,
                 patience: int = 20,
                 device: str = 'cpu',
                 random_state: int = 42):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self._model = None

    def _build_model(self, n_features: int) -> nn.Module:
        layers = []
        prev_dim = n_features
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _listnet_loss(scores: torch.Tensor,
                      y: torch.Tensor,
                      group_ids: torch.Tensor) -> torch.Tensor:
        """
        ListNet top-1 loss: for each group, KL between softmax(scores) and softmax(targets).
        Skips groups where all relevance is 0 (degenerate target distribution).
        """
        loss = scores.new_zeros(())
        n_valid = 0
        for g in torch.unique(group_ids):
            mask = group_ids == g
            s = scores[mask]
            t = y[mask]
            if t.sum() == 0 or s.numel() < 2:
                continue
            log_p = torch.log_softmax(s, dim=0)
            q = torch.softmax(t, dim=0)
            loss = loss - (q * log_p).sum()
            n_valid += 1
        if n_valid == 0:
            return loss
        return loss / n_valid

    def _encode_groups(self, groups: np.ndarray) -> torch.Tensor:
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        return torch.tensor(inverse, dtype=torch.long, device=self.device)

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            groups: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            eval_groups: Optional[np.ndarray] = None) -> 'MLPRanker':

        if groups is None:
            raise ValueError("MLPRanker requires groups for fit")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        groups_t = self._encode_groups(np.asarray(groups))

        self._model = self._build_model(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        has_val = eval_set is not None and eval_groups is not None
        if has_val:
            X_val, y_val = eval_set
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            val_groups_t = self._encode_groups(np.asarray(eval_groups))

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._model.train()
            optimizer.zero_grad()
            scores = self._model(X_t).squeeze(-1)
            loss = self._listnet_loss(scores, y_t, groups_t)
            loss.backward()
            optimizer.step()

            if has_val:
                self._model.eval()
                with torch.no_grad():
                    val_scores = self._model(X_val_t).squeeze(-1)
                    val_loss = self._listnet_loss(val_scores, y_val_t, val_groups_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone()
                                  for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Ranker has not been fit yet")
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self._model(X_t).squeeze(-1)
        return scores.cpu().numpy()
