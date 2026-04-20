"""
=============================================================
  Tahap 2 — Neuro-Fuzzy ANN (Backpropagation)
  Arsitektur mencerminkan struktur FIS Mamdani:
    Layer 1 (12 nodes) : Fuzzifikasi — satu node per MF term
                         (3 terms × 4 input variables)
    Layer 2 (30 nodes) : Rule Base   — satu node per rule
    Layer 3 (1 node)   : Defuzzifikasi — skor risiko 0–100

  Peran dalam sistem:
    • Dilatih untuk mereproduksi output Manual FIS pada dataset.
    • Bobot Layer 2 (rule_layer) yang terlatih merefleksikan
      tingkat kepentingan relatif setiap rule — inilah "tuning"-nya.
    • Bobot Layer 1 (mf_layer) mencerminkan sensitivitas
      setiap MF terhadap skor akhir.
    • Output ANN digunakan sebagai skor prediksi alternatif
      yang dibandingkan langsung dengan Manual FIS & GA FIS.
=============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  1. NETWORK ARCHITECTURE
# ─────────────────────────────────────────────

class NeuroFuzzyNet(nn.Module):
    """
    Three-layer network mirroring the Mamdani FIS structure.
    Input  : 4 normalised features [ipk, kehadiran, mk_gagal, status_ekon]
    Output : 1 scalar in [0, 100]  (risiko dropout score)
    """
    def __init__(self):
        super(NeuroFuzzyNet, self).__init__()
        # Layer 1: Fuzzification — 4 inputs → 12 MF nodes (3 per variable)
        self.mf_layer    = nn.Linear(4, 12)
        self.bn1         = nn.BatchNorm1d(12)
        # Layer 2: Rule base — 12 fuzz. activations → 30 rule nodes
        self.rule_layer  = nn.Linear(12, 30)
        self.bn2         = nn.BatchNorm1d(30)
        # Layer 3: Defuzzification — 30 rule strengths → 1 score
        self.output_layer = nn.Linear(30, 1)

        # Weight initialisation: small positive values mimic MF overlaps
        nn.init.xavier_uniform_(self.mf_layer.weight)
        nn.init.xavier_uniform_(self.rule_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Layer 1: sigmoid — bounded [0,1] like membership degrees
        x = torch.sigmoid(self.mf_layer(x))
        if x.shape[0] > 1:          # BatchNorm requires batch size > 1
            x = self.bn1(x)
        # Layer 2: relu — non-negative rule firing strengths
        x = torch.relu(self.rule_layer(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        # Layer 3: sigmoid scaled to [0, 100]
        return torch.sigmoid(self.output_layer(x)) * 100.0


# ─────────────────────────────────────────────
#  2. NORMALISATION HELPERS
# ─────────────────────────────────────────────

# Fixed normalisation ranges matching the FIS universe of discourse
_NORM = {
    'ipk':         (0.0, 4.0),
    'kehadiran':   (0.0, 100.0),
    'mk_gagal':    (0.0, 10.0),
    'status_ekon': (0.0, 1.0),
}

def _normalise(ipk_v, had_v, mk_v, eko_v):
    """Scale each input to [0, 1]."""
    return np.array([
        (ipk_v  - _NORM['ipk'][0])         / (_NORM['ipk'][1]         - _NORM['ipk'][0]),
        (had_v  - _NORM['kehadiran'][0])   / (_NORM['kehadiran'][1]   - _NORM['kehadiran'][0]),
        (mk_v   - _NORM['mk_gagal'][0])    / (_NORM['mk_gagal'][1]    - _NORM['mk_gagal'][0]),
        (eko_v  - _NORM['status_ekon'][0]) / (_NORM['status_ekon'][1] - _NORM['status_ekon'][0]),
    ], dtype=np.float32)


# ─────────────────────────────────────────────
#  3. DATASET PREPARATION
# ─────────────────────────────────────────────

def prepare_dataset(fis_sim, dataset):
    """
    Build (X, y) tensors from the UCI sample dataset.
    X : normalised 4-feature vectors
    y : FIS output score (0–100) — ANN learns to match this
    Only rows where the FIS successfully produces output are kept.
    """
    X_list, y_list = [], []
    for row in dataset:
        try:
            fis_sim.input['ipk']         = float(row['ipk'])
            fis_sim.input['kehadiran']   = float(row['kehadiran'])
            fis_sim.input['mk_gagal']    = float(row['mk_gagal'])
            fis_sim.input['status_ekon'] = float(row['status_ekon'])
            fis_sim.compute()
            score = float(fis_sim.output['risiko'])
            x_vec = _normalise(row['ipk'], row['kehadiran'],
                               row['mk_gagal'], row['status_ekon'])
            X_list.append(x_vec)
            y_list.append([score])
        except Exception:
            pass

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────
#  4. TRAINING LOOP
# ─────────────────────────────────────────────

def train_ann(fis_sim, dataset, epochs=150, lr=1e-3, batch_size=16,
              progress_callback=None):
    """
    Train NeuroFuzzyNet to replicate Manual FIS output.

    Args:
        fis_sim           : running ControlSystemSimulation (Manual FIS)
        dataset           : list of dicts from load_uci_sample()
        epochs            : number of training epochs
        lr                : learning rate for Adam
        batch_size        : mini-batch size
        progress_callback : optional callable(epoch, total_epochs, loss)
                            called after each epoch — use for Streamlit progress

    Returns:
        model       : trained NeuroFuzzyNet
        loss_history: list of mean MSE loss per epoch
        acc_history : list of classification accuracy per epoch
    """
    torch.manual_seed(42)
    X, y = prepare_dataset(fis_sim, dataset)

    loader = DataLoader(TensorDataset(X, y),
                        batch_size=batch_size, shuffle=True)

    model     = NeuroFuzzyNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    acc_history  = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # Classification accuracy on the full set
        model.eval()
        with torch.no_grad():
            all_preds = model(X).squeeze().numpy()
        labels = np.array([
            'Rendah' if s < 40 else ('Sedang' if s < 65 else 'Tinggi')
            for s in all_preds
        ])
        true_labels = np.array([
            'Rendah' if (s.item() < 40) else ('Sedang' if (s.item() < 65) else 'Tinggi')
            for s in y
        ])
        acc = float(np.mean(labels == true_labels))
        acc_history.append(acc)

        if progress_callback:
            progress_callback(epoch, epochs, mean_loss)

    model.eval()
    return model, loss_history, acc_history


# ─────────────────────────────────────────────
#  5. INFERENCE
# ─────────────────────────────────────────────

def predict_ann(model, ipk_v, had_v, mk_v, eko_v):
    """
    Run inference with a trained NeuroFuzzyNet.
    Returns (score: float, label: str).
    """
    model.eval()
    x_vec = _normalise(ipk_v, had_v, mk_v, eko_v)
    x_t   = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)  # [1, 4]
    with torch.no_grad():
        score = float(model(x_t).item())
    label = 'Rendah' if score < 40 else ('Sedang' if score < 65 else 'Tinggi')
    return round(score, 2), label


def evaluate_ann(model, dataset):
    """
    Evaluate trained ANN on the dataset.
    Returns same dict structure as fis_manual.evaluate() for consistency.
    """
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    correct  = 0
    confusion = {t: {p: 0 for p in label_order} for t in label_order}
    results   = []

    for row in dataset:
        try:
            score, pred = predict_ann(model, row['ipk'], row['kehadiran'],
                                      row['mk_gagal'], row['status_ekon'])
            true_l = row['label_true']
            confusion[true_l][pred] += 1
            if pred == true_l:
                correct += 1
            results.append({'true': true_l, 'pred': pred, 'score': score})
        except Exception:
            pass

    n        = len(results)
    accuracy = correct / n if n > 0 else 0

    per_class = {}
    for lbl in label_order:
        tp    = confusion[lbl][lbl]
        total = sum(confusion[lbl].values())
        per_class[lbl] = tp / total if total > 0 else 0

    return {
        'accuracy':  round(accuracy * 100, 2),
        'n':         n,
        'correct':   correct,
        'confusion': confusion,
        'per_class': per_class,
        'results':   results,
    }


def get_rule_weights(model):
    """
    Extract the absolute mean weight of each rule node from the trained model.
    Returns a numpy array of shape (30,) representing relative rule importance.
    """
    with torch.no_grad():
        # rule_layer.weight : [30, 12] — sum absolute weights across MF inputs
        importance = model.rule_layer.weight.abs().mean(dim=1).numpy()
    # Normalise to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance