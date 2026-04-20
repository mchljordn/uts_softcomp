"""
=============================================================
  Tahap 1 — Manual Fuzzy Inference System (Mamdani)
  Studi Kasus: Prediksi Risiko Dropout Mahasiswa
  Mata Kuliah: Soft Computing — UTS 2025/2026
  Dataset    : UCI #697 — Predict Students' Dropout and
               Academic Success (Realinho et al., 2022)
=============================================================
  Mapping Kolom UCI → Variabel FIS:

  ipk        ← rata-rata grade sem 1 & sem 2 (skala 0-20)
               dinormalisasi ke 0-4 (÷5)
  kehadiran  ← (approved_sem1 + approved_sem2) /
               max(enrolled_sem1 + enrolled_sem2, 1) x 100
  mk_gagal   ← clip((enrolled - approved) sem1 + sem2, 0, 10)
  status_ekon← Scholarship_holder (0 atau 1)

  Label UCI → FIS:
    'Graduate' → Rendah
    'Enrolled' → Sedang
    'Dropout'  → Tinggi

  FIS Type  : Mamdani
  AND       : Minimum
  OR        : Maximum
  Defuzz    : Centroid (Center of Gravity)
  Rules     : 31
=============================================================
  Instalasi:
    pip install scikit-fuzzy matplotlib numpy pandas ucimlrepo scipy
=============================================================
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════
#  1. MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════════════

def build_fis():
    """Bangun FIS Mamdani — return tuple variabel fuzzy."""

    ipk         = ctrl.Antecedent(np.arange(0, 4.01, 0.01), 'ipk')
    kehadiran   = ctrl.Antecedent(np.arange(0, 101,  1),    'kehadiran')
    mk_gagal    = ctrl.Antecedent(np.arange(0, 10.1, 0.1),  'mk_gagal')
    status_ekon = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'status_ekon')
    risiko      = ctrl.Consequent(np.arange(0, 101,  1),    'risiko',
                                  defuzzify_method='centroid')

    # ── IPK (0–4) ──
    ipk['Rendah'] = fuzz.trapmf(ipk.universe, [0.0, 0.0, 1.5, 2.2])
    ipk['Sedang'] = fuzz.trimf (ipk.universe, [1.8, 2.5, 3.2])
    ipk['Tinggi'] = fuzz.trapmf(ipk.universe, [2.8, 3.3, 4.0, 4.0])

    # ── Kehadiran (0–100%) ──
    kehadiran['Jarang'] = fuzz.trapmf(kehadiran.universe, [0,  0,  40, 60])
    kehadiran['Cukup']  = fuzz.trimf (kehadiran.universe, [50, 70, 85])
    kehadiran['Rajin']  = fuzz.trapmf(kehadiran.universe, [75, 85, 100, 100])

    # ── MK Gagal (0–10) ──
    mk_gagal['Sedikit'] = fuzz.trapmf(mk_gagal.universe, [0, 0, 1,  3])
    mk_gagal['Sedang']  = fuzz.trimf (mk_gagal.universe, [2, 4, 6])
    mk_gagal['Banyak']  = fuzz.trapmf(mk_gagal.universe, [5, 7, 10, 10])

    # ── Status Ekonomi (0=Rentan, 1=Stabil) — Gaussian ──
    status_ekon['Rentan'] = fuzz.gaussmf(status_ekon.universe, 0.0, 0.15)
    status_ekon['Stabil'] = fuzz.gaussmf(status_ekon.universe, 1.0, 0.15)

    # ── Output Risiko Dropout (0–100) ──
    risiko['Rendah'] = fuzz.trapmf(risiko.universe, [0,  0,  25, 45])
    risiko['Sedang'] = fuzz.trimf (risiko.universe, [35, 50, 65])
    risiko['Tinggi'] = fuzz.trapmf(risiko.universe, [55, 75, 100, 100])

    return ipk, kehadiran, mk_gagal, status_ekon, risiko


# ══════════════════════════════════════════════════════
#  2. RULE BASE — 31 Rules (Intuisi Pakar)
# ══════════════════════════════════════════════════════

def build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko):
    rules = [
        # ── RISIKO TINGGI (10 rules) ──
        ctrl.Rule(ipk['Rendah'] & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Rajin']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Tinggi']),

        # ── RISIKO SEDANG (13 rules) ──
        ctrl.Rule(ipk['Rendah'] & kehadiran['Cukup']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Cukup']  & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah'] & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Sedang']),
        # Menutup gap kombinasi "IPK tinggi + jarang hadir + MK gagal sedikit + ekonomi stabil"
        # yang sebelumnya bisa membuat tidak ada consequent aktif.
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Sedang']),

        # ── RISIKO RENDAH (8 rules) ──
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi'] & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang'] & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
    ]
    return rules


# ══════════════════════════════════════════════════════
#  3. PREDIKSI SINGLE INSTANCE
# ══════════════════════════════════════════════════════

def predict(sim, ipk_val, kehadiran_val, mk_gagal_val, status_ekon_val):
    """
    Inferensi satu mahasiswa.
    Return: (skor_float, label_str)
    """
    sim.input['ipk']         = float(np.clip(ipk_val,         0.0, 4.0))
    sim.input['kehadiran']   = float(np.clip(kehadiran_val,   0.0, 100.0))
    sim.input['mk_gagal']    = float(np.clip(mk_gagal_val,    0.0, 10.0))
    sim.input['status_ekon'] = float(np.clip(status_ekon_val, 0.0, 1.0))
    sim.compute()
    if 'risiko' not in sim.output:
        raise ValueError(
            "Inferensi gagal: tidak ada output 'risiko'. "
            f"Input={sim.input}. Kombinasi rule kemungkinan belum tercakup."
        )
    score = sim.output['risiko']
    label = 'Rendah' if score < 40 else ('Sedang' if score < 65 else 'Tinggi')
    return round(score, 2), label


# ══════════════════════════════════════════════════════
#  4. LOAD & PREPROCESS DATASET UCI #697
# ══════════════════════════════════════════════════════

def load_uci_dataset(sample_n=None, random_state=42):
    """
    Fetch dataset UCI #697 via ucimlrepo dan mapping ke variabel FIS.

    Args:
        sample_n     : int | None. Jika diisi, ambil stratified sample
                       sebanyak sample_n per kelas.
        random_state : seed numpy untuk reproducibility.

    Returns:
        list of dict: [{ipk, kehadiran, mk_gagal, status_ekon, label_true}, ...]
    """
    print("  [UCI] Mengambil dataset dari ucimlrepo (id=697)...")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "Library ucimlrepo belum terinstall.\n"
            "Jalankan: pip install ucimlrepo"
        )

    ds = fetch_ucirepo(id=697)
    X  = ds.data.features.copy()
    y  = ds.data.targets.copy()

    # Normalisasi nama kolom agar robust terhadap variasi format ucimlrepo:
    # contoh: "Curricular_units_1st_sem_(grade)" -> "curricular_units_1st_sem_grade"
    X.columns = (X.columns
                   .str.strip()
                   .str.replace(r'[^0-9a-zA-Z]+', '_', regex=True)
                   .str.replace(r'_+', '_', regex=True)
                   .str.strip('_')
                   .str.lower())

    print(f"  [UCI] Raw dataset : {len(X)} baris | {X.shape[1]} fitur")

    # ── Helper: cari kolom dari beberapa kandidat nama ──
    def find_col(df, candidates):
        col_lookup = {col.lower(): col for col in df.columns}
        for c in candidates:
            key = c.lower()
            if key in col_lookup:
                return col_lookup[key]

        for c in candidates:
            key = c.lower().replace('_', '')
            matches = [col for col in df.columns
                       if key in col.lower().replace('_', '')]
            if matches:
                return matches[0]
        return None

    col_grade1  = find_col(X, ['curricular_units_1st_sem_grade'])
    col_grade2  = find_col(X, ['curricular_units_2nd_sem_grade'])
    col_enr1    = find_col(X, ['curricular_units_1st_sem_enrolled'])
    col_enr2    = find_col(X, ['curricular_units_2nd_sem_enrolled'])
    col_appr1   = find_col(X, ['curricular_units_1st_sem_approved'])
    col_appr2   = find_col(X, ['curricular_units_2nd_sem_approved'])
    col_scholar = find_col(X, ['scholarship_holder'])

    # Validasi semua kolom ditemukan
    col_map = {
        'grade_sem1':    col_grade1,
        'grade_sem2':    col_grade2,
        'enrolled_sem1': col_enr1,
        'enrolled_sem2': col_enr2,
        'approved_sem1': col_appr1,
        'approved_sem2': col_appr2,
        'scholarship':   col_scholar,
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}\n"
            f"Kolom tersedia: {list(X.columns)}"
        )

    print(f"  [UCI] Mapping kolom berhasil:")
    for k, v in col_map.items():
        print(f"        {k:<15} ← {v}")

    # ── Feature Engineering ──
    df = pd.DataFrame()

    # IPK: rata-rata grade sem1+sem2, skala 0–20 dinormalisasi ke 0–4
    df['ipk'] = (
        (X[col_grade1].fillna(0) + X[col_grade2].fillna(0)) / 2 / 5
    ).clip(0, 4)

    # Kehadiran: rasio MK approved / enrolled × 100
    enrolled_total = (X[col_enr1].fillna(0) + X[col_enr2].fillna(0)).clip(lower=1)
    approved_total = (X[col_appr1].fillna(0) + X[col_appr2].fillna(0))
    df['kehadiran'] = (approved_total / enrolled_total * 100).clip(0, 100)

    # MK Gagal: (enrolled - approved) di kedua semester, clip 0–10
    mk_fail = (
        (X[col_enr1].fillna(0) - X[col_appr1].fillna(0)) +
        (X[col_enr2].fillna(0) - X[col_appr2].fillna(0))
    )
    df['mk_gagal'] = mk_fail.clip(0, 10)

    # Status Ekonomi: penerima beasiswa = 1 (Stabil), tidak = 0 (Rentan)
    df['status_ekon'] = X[col_scholar].fillna(0).astype(float)

    # Label
    label_col = y.columns[0]
    label_map = {'Graduate': 'Rendah', 'Enrolled': 'Sedang', 'Dropout': 'Tinggi'}
    df['label_true'] = y[label_col].map(label_map)

    # Bersihkan baris invalid
    df = df.dropna(subset=['label_true'])
    df = df[df['label_true'].isin(['Rendah', 'Sedang', 'Tinggi'])]

    print(f"\n  [UCI] Distribusi label setelah mapping:")
    for lbl, cnt in df['label_true'].value_counts().items():
        pct = cnt / len(df) * 100
        bar = '█' * int(pct / 3)
        print(f"        {lbl:<8}: {cnt:>5} sampel  ({pct:.1f}%)  {bar}")

    # ── Stratified sampling ──
    if sample_n is not None:
        groups = []
        for lbl in ['Rendah', 'Sedang', 'Tinggi']:
            subset = df[df['label_true'] == lbl]
            n      = min(sample_n, len(subset))
            groups.append(subset.sample(n=n, random_state=random_state))
        df = pd.concat(groups).sample(frac=1, random_state=random_state)
        print(f"\n  [UCI] Setelah stratified sampling (n_per_class={sample_n}): "
              f"{len(df)} baris total")

    # Statistik fitur
    print(f"\n  [UCI] Statistik fitur hasil mapping:")
    print(df[['ipk', 'kehadiran', 'mk_gagal', 'status_ekon']].describe().round(3).to_string())

    return df.to_dict('records')


# ══════════════════════════════════════════════════════
#  5. EVALUASI BATCH
# ══════════════════════════════════════════════════════

def evaluate(sim, dataset):
    """
    Jalankan inferensi pada seluruh dataset dan hitung metrik.

    Return dict:
        accuracy, macro_precision, macro_recall, macro_f1,
        per_class, per_class_full, confusion, n, skipped, results
    """
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    confusion   = {t: {p: 0 for p in label_order} for t in label_order}
    results     = []
    skipped     = 0
    total       = len(dataset)

    for i, row in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{total} diproses", end='\r')
        try:
            score, pred = predict(sim,
                                  row['ipk'], row['kehadiran'],
                                  row['mk_gagal'], row['status_ekon'])
            true_l = row['label_true']
            confusion[true_l][pred] += 1
            results.append({'true': true_l, 'pred': pred, 'score': score})
        except Exception:
            skipped += 1
    print()

    n        = len(results)
    correct  = sum(1 for r in results if r['true'] == r['pred'])
    accuracy = correct / n if n > 0 else 0

    # Per-class metrics
    per_class      = {}
    per_class_full = {}
    precisions, recalls, f1s = [], [], []

    for lbl in label_order:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[o][lbl] for o in label_order if o != lbl)
        fn = sum(confusion[lbl][o] for o in label_order if o != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc  = confusion[lbl][lbl] / max(sum(confusion[lbl].values()), 1)

        per_class[lbl]      = acc
        per_class_full[lbl] = {'accuracy': acc, 'precision': prec,
                                'recall': rec, 'f1': f1}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        'accuracy':        round(accuracy * 100, 2),
        'macro_precision': round(np.mean(precisions) * 100, 2),
        'macro_recall':    round(np.mean(recalls) * 100, 2),
        'macro_f1':        round(np.mean(f1s) * 100, 2),
        'per_class':       per_class,
        'per_class_full':  per_class_full,
        'confusion':       confusion,
        'n':               n,
        'skipped':         skipped,
        'results':         results,
    }


# ══════════════════════════════════════════════════════
#  6. VISUALISASI MEMBERSHIP FUNCTIONS
# ══════════════════════════════════════════════════════

def plot_membership_functions(ipk, kehadiran, mk_gagal, status_ekon, risiko,
                               save_path='mf_tahap1.png'):
    COLORS = {
        'Rendah': '#1D9E75', 'Sedang': '#EF9F27', 'Tinggi': '#E24B4A',
        'Jarang': '#E24B4A', 'Cukup':  '#EF9F27', 'Rajin':  '#1D9E75',
        'Sedikit':'#1D9E75', 'Banyak': '#E24B4A',
        'Rentan': '#E24B4A', 'Stabil': '#1D9E75',
    }

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Membership Functions — Tahap 1: Manual FIS (Mamdani)\n'
        'Prediksi Risiko Dropout Mahasiswa | Dataset UCI #697',
        fontsize=13, fontweight='bold', y=0.98
    )
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    vars_info = [
        (ipk,         'IPK Semester',           'Nilai IPK (0–4)'),
        (kehadiran,   'Tingkat Kehadiran',       'Kehadiran (%)'),
        (mk_gagal,    'Jumlah MK Gagal',         'Jumlah MK'),
        (status_ekon, 'Status Ekonomi',          'Encoded (0=Rentan, 1=Stabil)'),
        (risiko,      'Risiko Dropout (Output)', 'Skor Risiko (0–100)'),
    ]

    for ax, (var, title, xlabel) in zip(axes, vars_info):
        for label in var.terms:
            color = COLORS.get(label, '#888888')
            ax.plot(var.universe, var[label].mf,
                    label=label, color=color, linewidth=2.2)
            ax.fill_between(var.universe, var[label].mf, alpha=0.09, color=color)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('μ (derajat keanggotaan)', fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[5].axis('off')
    axes[5].text(
        0.5, 0.55,
        'FIS Type     : Mamdani\n'
        'AND operator : Minimum\n'
        'OR operator  : Maximum\n'
        'Defuzzifikasi: Centroid\n'
        'Total Rules  : 31\n\n'
        'Dataset : UCI #697\n'
        'Realinho et al. (2022)',
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.7',
                  facecolor='#f0f4ff', edgecolor='#aabbdd', linewidth=1.2),
        transform=axes[5].transAxes
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [OK] MF plot → {save_path}")
    return fig


# ══════════════════════════════════════════════════════
#  7. VISUALISASI EVALUASI
# ══════════════════════════════════════════════════════

def plot_evaluation(result, save_path='eval_tahap1.png'):
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    confusion   = result['confusion']
    cm_array    = np.array([[confusion[t][p] for p in label_order]
                             for t in label_order])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Evaluasi Tahap 1 — Manual FIS | Dataset UCI #697\n"
        f"Akurasi: {result['accuracy']}%  |  "
        f"F1 Macro: {result['macro_f1']}%  |  "
        f"n={result['n']}",
        fontsize=12, fontweight='bold'
    )

    # ── Confusion Matrix ──
    ax = axes[0]
    im = ax.imshow(cm_array, cmap='Blues', vmin=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(label_order)
    ax.set_yticks(range(3)); ax.set_yticklabels(label_order)
    ax.set_xlabel('Prediksi', fontweight='bold')
    ax.set_ylabel('Aktual',   fontweight='bold')
    ax.set_title('Confusion Matrix')
    for i in range(3):
        for j in range(3):
            v = cm_array[i, j]
            c = 'white' if v > cm_array.max() * 0.6 else 'black'
            ax.text(j, i, str(v), ha='center', va='center',
                    fontsize=12, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax)

    # ── Per-class Accuracy Bar ──
    ax2    = axes[1]
    colors = ['#1D9E75', '#EF9F27', '#E24B4A']
    vals   = [result['per_class'][l] * 100 for l in label_order]
    bars   = ax2.bar(label_order, vals, color=colors, width=0.5,
                     edgecolor='white', linewidth=1.2)
    ax2.set_ylim(0, 120)
    ax2.set_xlabel('Kelas Risiko', fontweight='bold')
    ax2.set_ylabel('Akurasi (%)', fontweight='bold')
    ax2.set_title('Akurasi Per Kelas')
    ax2.axhline(result['accuracy'], color='gray', linestyle='--',
                linewidth=1.2, label=f"Overall: {result['accuracy']}%")
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.1f}%", ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [OK] Evaluation plot → {save_path}")
    return fig


# ══════════════════════════════════════════════════════
#  8. MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  TAHAP 1 — Manual FIS: Prediksi Risiko Dropout Mahasiswa")
    print("  Dataset : UCI #697 (Realinho et al., 2022)")
    print("=" * 62)

    # ── Build FIS ──
    print("\n[1/4] Membangun FIS Mamdani...")
    ipk, kehadiran, mk_gagal, status_ekon, risiko = build_fis()
    rules    = build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko)
    fis_ctrl = ctrl.ControlSystem(rules)
    sim      = ctrl.ControlSystemSimulation(fis_ctrl)
    print(f"      {len(rules)} rules berhasil dimuat.")

    # ── Plot MF ──
    print("\n[2/4] Visualisasi Membership Functions...")
    plot_membership_functions(ipk, kehadiran, mk_gagal, status_ekon, risiko)

    # ── Demo Prediksi ──
    print("\n[3/4] Demo Prediksi (5 skenario representatif):")
    print("-" * 62)
    scenarios = [
        ("Mahasiswa Ideal",              3.8, 95,  0, 0.95),
        ("Risiko Sedang",                2.4, 68,  3, 0.40),
        ("Dropout Kritis",               1.1, 22,  8, 0.05),
        ("IPK Baik tapi Jarang Hadir",   3.2, 38,  2, 0.80),
        ("Ekonomi Lemah + IPK OK",       2.9, 80,  1, 0.10),
    ]
    for name, iv, hv, mf, ev in scenarios:
        score, label = predict(sim, iv, hv, mf, ev)
        bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
        print(f"  {name:<33} | IPK={iv} Had={hv}% MKF={mf} Eko={ev}")
        print(f"  → Skor: {score:6.2f}  [{bar}]  [{label}]\n")

    # ── Load & Evaluasi ──
    print("[4/4] Load Dataset UCI #697 & Evaluasi Batch...")
    # sample_n=200 → 200 per kelas (600 total), cepat ~1–2 menit
    # Set None untuk evaluasi semua ~4424 baris (~5–10 menit)
    dataset = load_uci_dataset(sample_n=200, random_state=42)
    print(f"\n  Menjalankan inferensi pada {len(dataset)} sampel...")
    result  = evaluate(sim, dataset)

    # ── Print Ringkasan ──
    print("\n" + "=" * 62)
    print("  BASELINE MANUAL FIS — HASIL EVALUASI TAHAP 1")
    print("  (Catat angka ini untuk perbandingan Tahap 2 GA & Tahap 3 ANN)")
    print("=" * 62)
    print(f"  Akurasi Keseluruhan  : {result['accuracy']}%")
    print(f"  Macro Precision      : {result['macro_precision']}%")
    print(f"  Macro Recall         : {result['macro_recall']}%")
    print(f"  Macro F1-Score       : {result['macro_f1']}%")
    print(f"  Sampel Valid         : {result['n']}")
    print(f"  Skipped (error FIS)  : {result['skipped']}")

    pf = result['per_class_full']
    print(f"\n  {'Kelas':<10} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*50}")
    for lbl in ['Rendah', 'Sedang', 'Tinggi']:
        d   = pf[lbl]
        bar = '█' * int(d['accuracy'] * 15)
        print(f"  {lbl:<10} {d['accuracy']*100:>8.1f}%"
              f"  {d['precision']*100:>8.1f}%"
              f"  {d['recall']*100:>6.1f}%"
              f"  {d['f1']*100:>6.1f}%  {bar}")

    print("\n  Confusion Matrix (baris=Aktual, kolom=Prediksi):")
    lo = ['Rendah', 'Sedang', 'Tinggi']
    print("             " + "  ".join(f"{l:>8}" for l in lo))
    for t in lo:
        row = f"  {t:<10} " + "  ".join(
            f"{result['confusion'][t][p]:>8}" for p in lo)
        print(row)

    plot_evaluation(result)

    print("\n  [SELESAI] Output: mf_tahap1.png | eval_tahap1.png")
    print("=" * 62)

    return sim, result


if __name__ == '__main__':
    sim, result = main()
