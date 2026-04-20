"""
=============================================================
  Tahap 3 — True ANFIS (Adaptive Neuro-Fuzzy Inference System)
  Referensi: Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based
             fuzzy inference system. IEEE Trans. SMC, 23(3), 665–685.

  Arsitektur (5 Layer ANFIS):
    Layer 1 : Fuzzifikasi   — Gaussian MF dengan parameter c, σ
                              yang DAPAT DILATIH (nn.Parameter)
    Layer 2 : Firing Strength — Produk fuzzy-AND per rule (30 rules)
    Layer 3 : Normalisasi   — Firing strength ternormalisasi
    Layer 4 : Konsekuen     — Weighted output per rule (Sugeno order-0)
                              dengan koefisien yang DAPAT DILATIH
    Layer 5 : Defuzzifikasi — Weighted sum → skor risiko 0–100

  Catatan Desain:
    • Output Sugeno order-0: setiap rule memiliki konstanta konsekuen
      yang dioptimasi backpropagation. Pendekatan ini adalah definisi
      ANFIS klasik (Jang 1993) dan lebih mudah dipertahankan secara
      akademis daripada centroid Mamdani yang tidak diferensiabel.
    • Parameter MF yang terlatih (c dan σ) merepresentasikan
      "pergeseran kurva" yang diminta dalam analisis komparatif laporan.
    • Rule topology identik dengan Manual FIS (Tahap 1) — hanya
      parameter MF dan bobot konsekuen yang dioptimasi.
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
#  KONSTANTA: INDEKS MF PER VARIABEL INPUT
#  IPK      : MF 0=Rendah  1=Sedang  2=Tinggi
#  Kehadiran: MF 3=Jarang  4=Cukup   5=Rajin
#  MK_Gagal : MF 6=Sedikit 7=Sedang  8=Banyak
#  StatusEkon:MF 9=Rentan 10=Stabil
# ─────────────────────────────────────────────
N_MF = 11   # total MF nodes di Layer 1

# Setiap rule = tuple (ipk_mf, had_mf, mk_mf, eko_mf) → indeks ke N_MF
# Urutan rule identik dengan fis_manual.py
RULE_ANTECEDENTS = [
    # ══ RISIKO TINGGI (rules 0–9) ══
    (0, 3, 8, 9),   # Rendah, Jarang, Banyak,  Rentan
    (0, 3, 8, 10),  # Rendah, Jarang, Banyak,  Stabil
    (0, 3, 7, 9),   # Rendah, Jarang, Sedang,  Rentan
    (0, 4, 8, 9),   # Rendah, Cukup,  Banyak,  Rentan
    (0, 3, 7, 10),  # Rendah, Jarang, Sedang,  Stabil
    (0, 4, 8, 10),  # Rendah, Cukup,  Banyak,  Stabil
    (1, 3, 8, 9),   # Sedang, Jarang, Banyak,  Rentan
    (0, 5, 8, 9),   # Rendah, Rajin,  Banyak,  Rentan
    (1, 3, 8, 10),  # Sedang, Jarang, Banyak,  Stabil
    (0, 3, 6, 9),   # Rendah, Jarang, Sedikit, Rentan
    # ══ RISIKO SEDANG (rules 10–21) ══
    (0, 4, 7, 10),  # Rendah, Cukup,  Sedang,  Stabil
    (1, 4, 7, 9),   # Sedang, Cukup,  Sedang,  Rentan
    (1, 3, 7, 10),  # Sedang, Jarang, Sedang,  Stabil
    (1, 3, 6, 9),   # Sedang, Jarang, Sedikit, Rentan
    (0, 5, 7, 10),  # Rendah, Rajin,  Sedang,  Stabil
    (0, 4, 6, 9),   # Rendah, Cukup,  Sedikit, Rentan
    (1, 4, 8, 10),  # Sedang, Cukup,  Banyak,  Stabil
    (1, 3, 7, 9),   # Sedang, Jarang, Sedang,  Rentan
    (2, 3, 7, 9),   # Tinggi, Jarang, Sedang,  Rentan
    (0, 5, 6, 9),   # Rendah, Rajin,  Sedikit, Rentan
    (1, 5, 7, 9),   # Sedang, Rajin,  Sedang,  Rentan
    (2, 4, 8, 9),   # Tinggi, Cukup,  Banyak,  Rentan
    # ══ RISIKO RENDAH (rules 22–29) ══
    (2, 5, 6, 10),  # Tinggi, Rajin,  Sedikit, Stabil
    (2, 5, 6, 9),   # Tinggi, Rajin,  Sedikit, Rentan
    (2, 4, 6, 10),  # Tinggi, Cukup,  Sedikit, Stabil
    (1, 5, 6, 10),  # Sedang, Rajin,  Sedikit, Stabil
    (2, 5, 7, 10),  # Tinggi, Rajin,  Sedang,  Stabil
    (1, 5, 6, 9),   # Sedang, Rajin,  Sedikit, Rentan
    (2, 4, 6, 9),   # Tinggi, Cukup,  Sedikit, Rentan
    (1, 4, 6, 10),  # Sedang, Cukup,  Sedikit, Stabil
]
N_RULES = len(RULE_ANTECEDENTS)   # = 30

# Inisialisasi c (center) Gaussian dari MF manual (Tahap 1)
# Urutan: IPK(R,S,T), Had(J,C,Ra), MK(Sd,Se,B), Eko(Re,St)
_INIT_C = torch.tensor([
    0.75, 2.5,  3.6,    # IPK:       Rendah≈0.75, Sedang≈2.5, Tinggi≈3.6
    30.0, 70.0, 90.0,   # Kehadiran: Jarang≈30,  Cukup≈70,   Rajin≈90
    0.5,  4.0,  8.5,    # MK_Gagal:  Sedikit≈0.5,Sedang≈4,   Banyak≈8.5
    0.0,  1.0,          # StatusEkon:Rentan≈0,   Stabil≈1
], dtype=torch.float32)

# Inisialisasi σ (sigma) — lebar kurva Gaussian
_INIT_S = torch.tensor([
    0.55, 0.50, 0.45,   # IPK
    18.0, 13.0, 10.0,   # Kehadiran
    0.80, 1.20, 1.50,   # MK_Gagal
    0.15, 0.15,         # StatusEkon
], dtype=torch.float32)

# Normalisasi ranges (sama dengan fis_manual universe of discourse)
_RANGES = torch.tensor([
    [0.0, 4.0],     # IPK
    [0.0, 100.0],   # Kehadiran
    [0.0, 10.0],    # MK_Gagal
    [0.0, 1.0],     # StatusEkon
], dtype=torch.float32)


# ─────────────────────────────────────────────
#  1. MODEL ANFIS
# ─────────────────────────────────────────────

class ANFISNet(nn.Module):
    """
    True 5-layer ANFIS dengan Gaussian MF yang dapat dilatih.

    Parameter yang dioptimasi backpropagation:
        self.c       : [N_MF]  — center setiap Gaussian MF
        self.log_s   : [N_MF]  — log(sigma), agar σ > 0 selalu terjaga
        self.conseq  : [N_RULES] — konstanta konsekuen Sugeno order-0

    Input  : batch × 4 (nilai asli, belum dinormalisasi)
    Output : batch × 1 (skor risiko 0–100)
    """

    def __init__(self):
        super(ANFISNet, self).__init__()

        # Layer 1 params: learnable Gaussian MF centers & log-sigmas
        self.c     = nn.Parameter(_INIT_C.clone())
        self.log_s = nn.Parameter(torch.log(_INIT_S.clone()))

        # Layer 4 params: one consequent constant per rule
        # Inisialisasi: Tinggi rules → ~75, Sedang rules → ~50, Rendah rules → ~20
        init_conseq = torch.zeros(N_RULES)
        init_conseq[0:10]  = 75.0   # Tinggi
        init_conseq[10:22] = 50.0   # Sedang
        init_conseq[22:30] = 20.0   # Rendah
        self.conseq = nn.Parameter(init_conseq)

        # Rule antecedent indices (fixed topology, not trained)
        ant = torch.tensor(RULE_ANTECEDENTS, dtype=torch.long)  # [30, 4]
        self.register_buffer('antecedents', ant)

    # ── MF index mapping: variable → MF node indices ──
    # IPK:       nodes 0,1,2   | input dim 0
    # Kehadiran: nodes 3,4,5   | input dim 1
    # MK_Gagal:  nodes 6,7,8   | input dim 2
    # StatusEkon:nodes 9,10    | input dim 3
    _VAR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _DIM_MAP  = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]  # which input dim feeds each MF node

    def forward(self, x):
        """
        x : [batch, 4] — raw input values (not normalised)
        """
        batch = x.shape[0]
        sigma = torch.exp(self.log_s).clamp(min=1e-3)  # ensure σ > 0

        # ── Layer 1: Fuzzification ──
        # For each MF node i, compute μ_i(x_dim) = exp(-0.5 * ((x - c_i)/σ_i)²)
        dim_map = torch.tensor(self._DIM_MAP, dtype=torch.long, device=x.device)
        x_sel   = x[:, dim_map]                       # [batch, N_MF]
        mu      = torch.exp(-0.5 * ((x_sel - self.c) / sigma) ** 2)  # [batch, N_MF]

        # ── Layer 2: Rule Firing Strength (product T-norm) ──
        # ant: [N_RULES, 4] → gather μ for each antecedent, multiply across 4 inputs
        ant_exp  = self.antecedents.unsqueeze(0).expand(batch, -1, -1)  # [batch,30,4]
        mu_exp   = mu.unsqueeze(1).expand(-1, N_RULES, -1)               # [batch,30,11]
        mu_rules = mu_exp.gather(2, ant_exp)                              # [batch,30,4]
        w        = mu_rules.prod(dim=2)                                   # [batch,30]

        # ── Layer 3: Normalised Firing Strengths ──
        w_sum  = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_norm = w / w_sum                                                # [batch,30]

        # ── Layer 4 & 5: Weighted Consequents → scalar output ──
        # Clamp conseq to [0, 100] to match output domain
        conseq_clamped = self.conseq.clamp(0.0, 100.0)
        out = (w_norm * conseq_clamped).sum(dim=1, keepdim=True)          # [batch,1]

        return out   # [batch, 1], range ≈ 0–100


# ─────────────────────────────────────────────
#  2. DATASET
# ─────────────────────────────────────────────

def prepare_dataset(fis_sim, dataset):
    """
    Buat tensor (X, y) dari dataset.
    X : nilai input RAW (4 fitur, tidak dinormalisasi) — ANFIS menerima nilai asli
    y : skor output Manual FIS (0–100) — target yang ingin direproduksi
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
            X_list.append([row['ipk'], row['kehadiran'],
                           row['mk_gagal'], row['status_ekon']])
            y_list.append([score])
        except Exception:
            pass

    X = torch.tensor(np.array(X_list, dtype=np.float32))
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y


# ─────────────────────────────────────────────
#  3. TRAINING
# ─────────────────────────────────────────────

def train_anfis(fis_sim, dataset, epochs=200, lr=1e-2, batch_size=16,
                progress_callback=None):
    """
    Latih ANFISNet untuk mereproduksi output Manual FIS.

    Args:
        fis_sim           : ControlSystemSimulation dari fis_manual.py
        dataset           : list of dicts dari load_uci_sample()
        epochs            : jumlah epoch pelatihan
        lr                : learning rate Adam
        batch_size        : ukuran mini-batch
        progress_callback : callable(epoch, total, loss) opsional

    Returns:
        model        : ANFISNet terlatih
        loss_history : list MSE per epoch
        acc_history  : list akurasi klasifikasi per epoch
        mf_before    : dict parameter MF sebelum pelatihan
        mf_after     : dict parameter MF setelah pelatihan
    """
    torch.manual_seed(42)
    X, y = prepare_dataset(fis_sim, dataset)

    model = ANFISNet()

    # Simpan parameter MF sebelum training untuk analisis pergeseran kurva
    mf_before = _extract_mf_params(model)

    loader    = DataLoader(TensorDataset(X, y),
                           batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
            # Gradient clipping untuk stabilitas
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # Akurasi klasifikasi
        model.eval()
        with torch.no_grad():
            all_preds = model(X).squeeze().numpy()
        pred_labels = _scores_to_labels(all_preds)
        true_labels = _scores_to_labels(y.squeeze().numpy())
        acc = float(np.mean(pred_labels == true_labels))
        acc_history.append(acc)

        if progress_callback:
            progress_callback(epoch, epochs, mean_loss)

    model.eval()
    mf_after = _extract_mf_params(model)

    return model, loss_history, acc_history, mf_before, mf_after


# ─────────────────────────────────────────────
#  4. INFERENSI
# ─────────────────────────────────────────────

def predict_anfis(model, ipk_v, had_v, mk_v, eko_v):
    """
    Inferensi satu sampel dengan ANFISNet terlatih.
    Return: (score: float, label: str)
    """
    model.eval()
    x = torch.tensor([[ipk_v, had_v, mk_v, eko_v]], dtype=torch.float32)
    with torch.no_grad():
        score = float(model(x).item())
    score = float(np.clip(score, 0.0, 100.0))
    label = 'Rendah' if score < 40 else ('Sedang' if score < 65 else 'Tinggi')
    return round(score, 2), label


def evaluate_anfis(model, dataset):
    """
    Evaluasi ANFISNet pada dataset.
    Return dict yang konsisten dengan fis_manual.evaluate().
    """
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    correct   = 0
    confusion = {t: {p: 0 for p in label_order} for t in label_order}
    results   = []

    for row in dataset:
        try:
            score, pred = predict_anfis(model, row['ipk'], row['kehadiran'],
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


# ─────────────────────────────────────────────
#  5. UTILITAS ANALISIS PERGESERAN MF
#     (untuk laporan & visualisasi UI)
# ─────────────────────────────────────────────

_MF_NAMES = [
    ('IPK',        'Rendah'),
    ('IPK',        'Sedang'),
    ('IPK',        'Tinggi'),
    ('Kehadiran',  'Jarang'),
    ('Kehadiran',  'Cukup'),
    ('Kehadiran',  'Rajin'),
    ('MK_Gagal',   'Sedikit'),
    ('MK_Gagal',   'Sedang'),
    ('MK_Gagal',   'Banyak'),
    ('StatusEkon', 'Rentan'),
    ('StatusEkon', 'Stabil'),
]

def _extract_mf_params(model):
    """Ekstrak c dan σ dari model sebagai dict yang mudah dibaca."""
    with torch.no_grad():
        c_vals = model.c.detach().numpy().copy()
        s_vals = torch.exp(model.log_s).detach().numpy().copy()
    params = {}
    for i, (var, term) in enumerate(_MF_NAMES):
        if var not in params:
            params[var] = {}
        params[var][term] = {'c': float(c_vals[i]), 'sigma': float(s_vals[i])}
    return params


def get_mf_shift_report(mf_before, mf_after):
    """
    Buat laporan pergeseran parameter MF sebelum vs sesudah training.
    Return: list of dicts dengan kolom Variabel, Term, c_before, c_after,
            delta_c, sigma_before, sigma_after, delta_sigma
    Berguna untuk tabel dalam laporan dan visualisasi di UI.
    """
    rows = []
    for var in mf_before:
        for term in mf_before[var]:
            b = mf_before[var][term]
            a = mf_after[var][term]
            rows.append({
                'Variabel':     var,
                'Term':         term,
                'c_before':     round(b['c'],     4),
                'c_after':      round(a['c'],     4),
                'delta_c':      round(a['c']      - b['c'],     4),
                'sigma_before': round(b['sigma'], 4),
                'sigma_after':  round(a['sigma'], 4),
                'delta_sigma':  round(a['sigma']  - b['sigma'], 4),
            })
    return rows


def get_gaussian_mf(universe, c, sigma):
    """
    Hitung nilai Gaussian MF pada array universe.
    Berguna untuk plot perbandingan MF sebelum vs sesudah di UI.
    """
    return np.exp(-0.5 * ((universe - c) / max(sigma, 1e-3)) ** 2)


def get_rule_weights(model):
    """
    Ekstrak bobot konsekuen per rule yang telah dilatih.
    Return: np.ndarray shape (30,) — nilai consequent setiap rule.
    Berguna untuk visualisasi kepentingan relatif setiap rule.
    """
    with torch.no_grad():
        return model.conseq.detach().numpy().copy()


# ─────────────────────────────────────────────
#  6. HELPER
# ─────────────────────────────────────────────

def _scores_to_labels(scores):
    return np.array([
        'Rendah' if s < 40 else ('Sedang' if s < 65 else 'Tinggi')
        for s in scores
    ])
