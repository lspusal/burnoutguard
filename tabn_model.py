"""
TABN - Temporal Attention Burnout Network

Arquitectura novel para predicción de burnout académico con:
1. BiLSTM para modelado temporal secuencial
2. Mecanismo de Atención Temporal (contribución principal)
3. Multi-Scale Temporal Pooling
4. Burnout-Aware Loss Function

Autor: Research Project
Versión: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """
    Mecanismo de Atención Temporal para Burnout
    
    Aprende a identificar qué momentos temporales son más relevantes
    para predecir burnout. Produce attention weights interpretables.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        self.attention_size = attention_size
        
        # Proyección a espacio de atención
        self.W = nn.Linear(hidden_size, attention_size, bias=False)
        self.b = nn.Parameter(torch.zeros(attention_size))
        
        # Vector de contexto learnable
        self.u = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_size]
            mask: [batch, seq_len] - 1 for valid, 0 for padding
            
        Returns:
            context: [batch, hidden_size] - weighted sum
            attention_weights: [batch, seq_len] - interpretable weights
        """
        # Score = tanh(W * h + b)
        score = torch.tanh(self.W(lstm_output) + self.b)  # [batch, seq, attn_size]
        
        # Attention weights = softmax(u * score)
        attention_score = self.u(score).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_score, dim=1)  # [batch, seq_len]
        
        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, attention_weights


class MultiScalePooling(nn.Module):
    """
    Multi-Scale Temporal Pooling
    
    Captura patrones a diferentes escalas temporales:
    - Local (últimos días)
    - Semanal (promedio por semana)
    - Global (todo el período)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Pooling adaptativos
        self.local_pool = nn.AdaptiveAvgPool1d(1)  # Último estado
        self.week_pool = nn.AdaptiveAvgPool1d(4)   # 4 "semanas"
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Todo
        
        # Combinar escalas
        self.combine = nn.Linear(hidden_size * 3, hidden_size)
        
    def forward(self, lstm_output: torch.Tensor):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_size]
            
        Returns:
            multi_scale: [batch, hidden_size]
        """
        # Transpose for pooling: [batch, hidden, seq]
        x = lstm_output.transpose(1, 2)
        
        # Multi-scale pooling
        local = lstm_output[:, -1, :]  # Último timestep [batch, hidden]
        week = self.week_pool(x).mean(dim=2)  # [batch, hidden]
        global_feat = self.global_pool(x).squeeze(-1)  # [batch, hidden]
        
        # Concatenate and combine
        combined = torch.cat([local, week, global_feat], dim=1)
        return self.combine(combined)


class TABN(nn.Module):
    """
    Temporal Attention Burnout Network
    
    Arquitectura:
    - Input Projection + Normalization
    - Bidirectional LSTM
    - Temporal Attention (contribución principal)
    - Multi-Scale Pooling
    - Classification Head
    
    Produces both predictions and interpretable attention weights.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 attention_size: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output is 2*hidden_size due to bidirectional
        lstm_output_size = hidden_size * 2
        
        # Temporal Attention (CONTRIBUCIÓN 1)
        self.attention = TemporalAttention(lstm_output_size, attention_size)
        
        # Multi-Scale Pooling (CONTRIBUCIÓN 2)
        self.multi_scale = MultiScalePooling(lstm_output_size)
        
        # Combine attention and multi-scale
        self.combine = nn.Linear(lstm_output_size * 2, hidden_size)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch, seq_len, input_size] - temporal sequence
            mask: [batch, seq_len] - 1 for valid, 0 for padding
            
        Returns:
            logits: [batch, 1] - burnout probability (before sigmoid)
            attention_weights: [batch, seq_len] - interpretable weights
        """
        # Input projection
        x = self.input_norm(x)
        x = F.relu(self.input_proj(x))
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Temporal Attention
        attn_context, attention_weights = self.attention(lstm_out, mask)
        
        # Multi-Scale Pooling
        multi_scale_context = self.multi_scale(lstm_out)
        
        # Combine both representations
        combined = torch.cat([attn_context, multi_scale_context], dim=1)
        combined = F.relu(self.combine(combined))
        
        # Classification
        x = self.dropout(combined)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attention_weights
    
    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get probability predictions"""
        logits, attention = self.forward(x, mask)
        proba = torch.sigmoid(logits)
        return proba, attention


class BurnoutAwareLoss(nn.Module):
    """
    Burnout-Aware Loss Function (CONTRIBUCIÓN 3)
    
    BCE loss con penalización adicional por:
    1. Falsos negativos tempranos (perderse burnout en primeras semanas)
    2. Calibración de probabilidades
    """
    
    def __init__(self, 
                 early_penalty_weight: float = 2.0,
                 class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.early_penalty_weight = early_penalty_weight
        self.class_weight = class_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                week_indices: Optional[torch.Tensor] = None):
        """
        Args:
            logits: [batch, 1] - model output
            targets: [batch] - true labels
            week_indices: [batch] - week number for each sample (0-indexed)
            
        Returns:
            loss: scalar
        """
        targets = targets.float().unsqueeze(1)
        
        # Base BCE loss
        base_loss = self.bce(logits, targets)
        
        # Apply class weights if provided
        if self.class_weight is not None:
            weights = torch.where(targets == 1, 
                                  self.class_weight[1], 
                                  self.class_weight[0])
            base_loss = base_loss * weights
        
        # Early detection penalty: penalize FN in early weeks
        if week_indices is not None:
            # Higher penalty for missing burnout in weeks 0-4
            early_mask = (week_indices < 4).float().unsqueeze(1)
            # Only penalize false negatives (pred=0, true=1)
            fn_mask = (torch.sigmoid(logits) < 0.5).float() * targets
            
            early_fn_penalty = early_mask * fn_mask * self.early_penalty_weight
            base_loss = base_loss + early_fn_penalty
        
        return base_loss.mean()


class TemporalDataset(Dataset):
    """Dataset para secuencias temporales de OULAD"""
    
    def __init__(self, 
                 sequences: List[np.ndarray],
                 labels: np.ndarray,
                 max_len: int = 30):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad or truncate to max_len
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]  # Keep last max_len timesteps
        elif len(seq) < self.max_len:
            padding = np.zeros((self.max_len - len(seq), seq.shape[1]))
            seq = np.vstack([padding, seq])
        
        # Create mask (1 for valid, 0 for padding)
        mask = np.ones(self.max_len)
        if len(self.sequences[idx]) < self.max_len:
            mask[:self.max_len - len(self.sequences[idx])] = 0
        
        return (
            torch.FloatTensor(seq),
            torch.FloatTensor(mask),
            torch.FloatTensor([label])
        )


def create_temporal_sequences(df_vle: pd.DataFrame, 
                               df_info: pd.DataFrame,
                               window_days: int = 7) -> Tuple[List, np.ndarray]:
    """
    Convertir datos OULAD en secuencias temporales para TABN
    
    Args:
        df_vle: Student VLE interactions
        df_info: Student info with final_result
        window_days: Días por ventana temporal
        
    Returns:
        sequences: List of [timesteps, features] arrays
        labels: Array of burnout labels
    """
    logger.info("Creando secuencias temporales para TABN...")
    
    # Target variable
    df_info['burnout'] = df_info['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
    
    # Aggregate VLE by week
    df_vle['week'] = df_vle['date'] // window_days
    
    weekly = df_vle.groupby(['id_student', 'code_module', 'code_presentation', 'week']).agg({
        'sum_click': ['sum', 'mean', 'std', 'max', 'count']
    }).reset_index()
    
    weekly.columns = ['id_student', 'code_module', 'code_presentation', 'week',
                      'total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks', 'n_events']
    weekly = weekly.fillna(0)
    
    # Create sequences per student-course
    sequences = []
    labels = []
    student_ids = []
    
    for (sid, module, pres), group in weekly.groupby(['id_student', 'code_module', 'code_presentation']):
        # Sort by week
        group = group.sort_values('week')
        
        # Feature columns
        features = group[['total_clicks', 'avg_clicks', 'std_clicks', 'max_clicks', 'n_events']].values
        
        # Add temporal features
        if len(features) > 1:
            # Week-over-week changes
            changes = np.diff(features[:, 0])  # Click changes
            changes = np.concatenate([[0], changes])
            features = np.column_stack([features, changes])
        else:
            features = np.column_stack([features, np.zeros(len(features))])
        
        # Get label
        student_info = df_info[
            (df_info['id_student'] == sid) & 
            (df_info['code_module'] == module) &
            (df_info['code_presentation'] == pres)
        ]
        
        if len(student_info) == 0:
            continue
            
        label = student_info['burnout'].values[0]
        
        sequences.append(features)
        labels.append(label)
        student_ids.append((sid, module, pres))
    
    logger.info(f"  ✓ Creadas {len(sequences)} secuencias")
    logger.info(f"  ✓ Burnout rate: {np.mean(labels)*100:.1f}%")
    
    return sequences, np.array(labels), student_ids


class TABNTrainer:
    """Trainer para TABN con early stopping y métricas"""
    
    def __init__(self,
                 model: TABN,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            sequences, masks, labels = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            logits, _ = self.model(sequences, masks)
            loss = self.criterion(logits, labels.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences, masks, labels = [b.to(self.device) for b in batch]
                
                logits, _ = self.model(sequences, masks)
                loss = self.criterion(logits, labels.squeeze())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                
                total_loss += loss.item()
                all_probs.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(dataloader)
        auroc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, auroc
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            patience: int = 10) -> Dict:
        """Train with early stopping"""
        
        best_auroc = 0
        patience_counter = 0
        best_state = None
        
        logger.info(f"Training TABN for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auroc = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auroc'].append(val_auroc)
            
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        logger.info(f"  ✓ Best validation AUROC: {best_auroc:.4f}")
        
        return self.history
    
    def get_attention_weights(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract attention weights for interpretability"""
        self.model.eval()
        all_attention = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences, masks, labels = [b.to(self.device) for b in batch]
                _, attention = self.model(sequences, masks)
                
                all_attention.extend(attention.cpu().numpy())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        return np.array(all_attention), np.array(all_labels)
