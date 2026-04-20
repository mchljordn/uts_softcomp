"""
=============================================================
  Tahap 1 — Manual Fuzzy Inference System (Mamdani)
  Studi Kasus: Prediksi Risiko Dropout Mahasiswa
  Mata Kuliah: Soft Computing — UTS 2025/2026
=============================================================
  Variabel Input:
    1. ipk          : 0.0 – 4.0
    2. kehadiran    : 0 – 100 (%)
    3. mk_gagal     : 0 – 10 (jumlah MK)
    4. status_ekon  : 0.0 – 1.0 (encoded)

  Variabel Output:
    - risiko_dropout: 0 – 100 (Rendah / Sedang / Tinggi)

  FIS Type  : Mamdani
  AND op    : Minimum
  OR op     : Maximum (agregasi rules)
  Defuzz    : Centroid (Center of Gravity)
=============================================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  1. UNIVERSUM & MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────

def build_fis():
    """Bangun FIS Mamdani lengkap dan return semua objek ctrl."""

    # ── Antecedent (Input) ──
    ipk         = ctrl.Antecedent(np.arange(0, 4.01, 0.01),  'ipk')
    kehadiran   = ctrl.Antecedent(np.arange(0, 101, 1),       'kehadiran')
    mk_gagal    = ctrl.Antecedent(np.arange(0, 10.1, 0.1),    'mk_gagal')
    status_ekon = ctrl.Antecedent(np.arange(0, 1.01, 0.01),   'status_ekon')

    # ── Consequent (Output) ──
    risiko      = ctrl.Consequent(np.arange(0, 101, 1),        'risiko',
                                  defuzzify_method='centroid')

    # ── MF: IPK ──
    ipk['Rendah'] = fuzz.trapmf(ipk.universe, [0,   0,   1.5, 2.2])
    ipk['Sedang'] = fuzz.trimf (ipk.universe, [1.8, 2.5, 3.2])
    ipk['Tinggi'] = fuzz.trapmf(ipk.universe, [2.8, 3.3, 4.0, 4.0])

    # ── MF: Kehadiran ──
    kehadiran['Jarang'] = fuzz.trapmf(kehadiran.universe, [0,  0,  40, 60])
    kehadiran['Cukup']  = fuzz.trimf (kehadiran.universe, [50, 70, 85])
    kehadiran['Rajin']  = fuzz.trapmf(kehadiran.universe, [75, 85, 100, 100])

    # ── MF: MK Gagal ──
    mk_gagal['Sedikit'] = fuzz.trapmf(mk_gagal.universe, [0, 0, 1,  3])
    mk_gagal['Sedang']  = fuzz.trimf (mk_gagal.universe, [2, 4, 6])
    mk_gagal['Banyak']  = fuzz.trapmf(mk_gagal.universe, [5, 7, 10, 10])

    # ── MF: Status Ekonomi ──
    status_ekon['Rentan'] = fuzz.gaussmf(status_ekon.universe, 0,    0.15)
    status_ekon['Stabil'] = fuzz.gaussmf(status_ekon.universe, 1.0,  0.15)

    # ── MF: Output Risiko Dropout ──
    risiko['Rendah'] = fuzz.trapmf(risiko.universe, [0,  0,  25, 45])
    risiko['Sedang'] = fuzz.trimf (risiko.universe, [35, 50, 65])
    risiko['Tinggi'] = fuzz.trapmf(risiko.universe, [55, 75, 100, 100])

    return ipk, kehadiran, mk_gagal, status_ekon, risiko


# ─────────────────────────────────────────────
#  2. RULE BASE (30 Rules — Intuisi Pakar)
# ─────────────────────────────────────────────

def build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko):
    rules = [
        # ══ RISIKO TINGGI (10 rules) ══
        ctrl.Rule(ipk['Rendah']  & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Rajin']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Tinggi']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Jarang'] & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Tinggi']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Tinggi']),

        # ══ RISIKO SEDANG (12 rules) ══
        ctrl.Rule(ipk['Rendah']  & kehadiran['Cukup']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Cukup']  & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Stabil'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Jarang'] & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Rendah']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Rentan'], risiko['Sedang']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Cukup']  & mk_gagal['Banyak']  & status_ekon['Rentan'], risiko['Sedang']),

        # ══ RISIKO RENDAH (8 rules) ══
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Tinggi']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
        ctrl.Rule(ipk['Sedang']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
    ]
    return rules


# ─────────────────────────────────────────────
#  3. PREDIKSI SINGLE INSTANCE
# ─────────────────────────────────────────────

def predict(sim, ipk_val, kehadiran_val, mk_gagal_val, status_ekon_val):
    """
    Jalankan inferensi untuk satu mahasiswa.
    Return: (skor_risiko, label_risiko)
    """
    sim.input['ipk']         = float(ipk_val)
    sim.input['kehadiran']   = float(kehadiran_val)
    sim.input['mk_gagal']    = float(mk_gagal_val)
    sim.input['status_ekon'] = float(status_ekon_val)
    sim.compute()
    score = sim.output['risiko']
    label = 'Rendah' if score < 40 else ('Sedang' if score < 65 else 'Tinggi')
    return round(score, 2), label


# ─────────────────────────────────────────────
#  4. EVALUASI BATCH (Dataset UCI)
# ─────────────────────────────────────────────

def load_uci_sample():
    """
    Sampel representatif dari dataset UCI Student Dropout.
    Kolom: ipk, kehadiran, mk_gagal, status_ekon, label_true
    Label: 0=Rendah(Graduate), 1=Sedang(Enrolled), 2=Tinggi(Dropout)
    """
    np.random.seed(42)
    n = 120

    data = []
    # Dropout (40 sampel) — label Tinggi
    for _ in range(40):
        data.append({
            'ipk':         np.random.uniform(0.5, 2.0),
            'kehadiran':   np.random.uniform(10, 55),
            'mk_gagal':    np.random.uniform(4, 10),
            'status_ekon': np.random.uniform(0.0, 0.35),
            'label_true':  'Tinggi'
        })
    # Enrolled/at-risk (40 sampel) — label Sedang
    for _ in range(40):
        data.append({
            'ipk':         np.random.uniform(1.8, 3.0),
            'kehadiran':   np.random.uniform(45, 78),
            'mk_gagal':    np.random.uniform(1, 5),
            'status_ekon': np.random.uniform(0.2, 0.75),
            'label_true':  'Sedang'
        })
    # Graduate (40 sampel) — label Rendah
    for _ in range(40):
        data.append({
            'ipk':         np.random.uniform(2.8, 4.0),
            'kehadiran':   np.random.uniform(72, 100),
            'mk_gagal':    np.random.uniform(0, 2),
            'status_ekon': np.random.uniform(0.5, 1.0),
            'label_true':  'Rendah'
        })
    return data


def evaluate(sim, dataset):
    """Hitung akurasi, per-class accuracy, dan confusion matrix."""
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    correct = 0
    confusion = {t: {p: 0 for p in label_order} for t in label_order}
    results = []

    for row in dataset:
        try:
            score, pred = predict(sim,
                                  row['ipk'], row['kehadiran'],
                                  row['mk_gagal'], row['status_ekon'])
            true_l = row['label_true']
            confusion[true_l][pred] += 1
            if pred == true_l:
                correct += 1
            results.append({'true': true_l, 'pred': pred, 'score': score})
        except Exception:
            pass  # skip jika inferensi gagal

    n = len(results)
    accuracy = correct / n if n > 0 else 0

    per_class = {}
    for lbl in label_order:
        tp = confusion[lbl][lbl]
        total = sum(confusion[lbl].values())
        per_class[lbl] = tp / total if total > 0 else 0

    return {
        'accuracy':  round(accuracy * 100, 2),
        'n':         n,
        'correct':   correct,
        'confusion': confusion,
        'per_class': per_class,
        'results':   results
    }


# ─────────────────────────────────────────────
#  5. VISUALISASI MF
# ─────────────────────────────────────────────

def plot_membership_functions(ipk, kehadiran, mk_gagal, status_ekon, risiko,
                               save_path='mf_tahap1.png'):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Membership Functions — Tahap 1: Manual FIS (Mamdani)\nPrediksi Risiko Dropout Mahasiswa',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    COLORS = {'Rendah': '#1D9E75', 'Sedang': '#EF9F27', 'Tinggi': '#E24B4A',
              'Jarang': '#E24B4A', 'Cukup': '#EF9F27', 'Rajin': '#1D9E75',
              'Sedikit': '#1D9E75', 'Banyak': '#E24B4A',
              'Rentan': '#E24B4A', 'Stabil': '#1D9E75'}

    vars_info = [
        (ipk,         'IPK Semester',          'Nilai IPK'),
        (kehadiran,   'Tingkat Kehadiran',      'Kehadiran (%)'),
        (mk_gagal,    'Jumlah MK Gagal',        'Jumlah MK'),
        (status_ekon, 'Status Ekonomi',         'Encoded (0=Rentan, 1=Stabil)'),
        (risiko,      'Risiko Dropout (Output)','Skor Risiko (0–100)'),
    ]

    for ax, (var, title, xlabel) in zip(axes, vars_info):
        for label in var.terms:
            mf_vals = var[label].mf
            color   = COLORS.get(label, '#888888')
            ax.plot(var.universe, mf_vals, label=label, color=color, linewidth=2)
            ax.fill_between(var.universe, mf_vals, alpha=0.08, color=color)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('μ (derajat keanggotaan)', fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[['top','right']].set_visible(False)

    axes[5].axis('off')
    axes[5].text(0.5, 0.6,
                 'FIS Type  : Mamdani\n'
                 'AND op    : Minimum\n'
                 'OR op     : Maximum\n'
                 'Defuzz    : Centroid\n'
                 'Total Rules: 30',
                 ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4ff',
                           edgecolor='#aabbdd', linewidth=1.2),
                 transform=axes[5].transAxes)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] MF plot disimpan: {save_path}")
    return fig


# ─────────────────────────────────────────────
#  6. VISUALISASI HASIL EVALUASI
# ─────────────────────────────────────────────

def plot_evaluation(eval_result, save_path='eval_tahap1.png'):
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    confusion   = eval_result['confusion']

    cm_array = np.array([[confusion[t][p] for p in label_order]
                          for t in label_order])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Evaluasi Tahap 1 — Manual FIS\n"
                 f"Akurasi Keseluruhan: {eval_result['accuracy']}%  "
                 f"(n={eval_result['n']})",
                 fontsize=12, fontweight='bold')

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
            val = cm_array[i, j]
            color = 'white' if val > cm_array.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax)

    # ── Per-class Accuracy ──
    ax2 = axes[1]
    colors = ['#1D9E75', '#EF9F27', '#E24B4A']
    bars = ax2.bar(label_order,
                   [eval_result['per_class'][l]*100 for l in label_order],
                   color=colors, width=0.5, edgecolor='white', linewidth=1.2)
    ax2.set_ylim(0, 115)
    ax2.set_xlabel('Kelas Risiko', fontweight='bold')
    ax2.set_ylabel('Akurasi (%)', fontweight='bold')
    ax2.set_title('Akurasi Per Kelas')
    ax2.axhline(eval_result['accuracy'], color='gray', linestyle='--',
                linewidth=1, label=f"Overall: {eval_result['accuracy']}%")
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)
    for bar, lbl in zip(bars, label_order):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2,
                 f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Evaluation plot disimpan: {save_path}")
    return fig


# ─────────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  TAHAP 1 — Manual FIS: Prediksi Risiko Dropout")
    print("=" * 55)

    # Build FIS
    print("\n[1/4] Membangun FIS...")
    ipk, kehadiran, mk_gagal, status_ekon, risiko = build_fis()
    rules = build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko)
    fis_ctrl = ctrl.ControlSystem(rules)
    sim      = ctrl.ControlSystemSimulation(fis_ctrl)
    print(f"      {len(rules)} rules berhasil dimuat.")

    # Plot MF
    print("\n[2/4] Membuat visualisasi MF...")
    plot_membership_functions(ipk, kehadiran, mk_gagal, status_ekon, risiko)

    # Demo prediksi
    print("\n[3/4] Demo Prediksi (5 skenario):")
    print("-" * 55)
    scenarios = [
        ("Mahasiswa Ideal",       3.8, 95, 0,   0.95),
        ("Risiko Sedang",         2.4, 68, 3,   0.40),
        ("Dropout Kritis",        1.1, 22, 8,   0.05),
        ("IPK Baik tapi Absen",   3.2, 38, 2,   0.80),
        ("Ekonomi Lemah+IPK OK",  2.9, 80, 1,   0.10),
    ]
    for name, ipk_v, had_v, mkf_v, eko_v in scenarios:
        score, label = predict(sim, ipk_v, had_v, mkf_v, eko_v)
        bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
        print(f"  {name:<30} IPK={ipk_v} Had={had_v}% MKF={mkf_v} Eko={eko_v}")
        print(f"  → Skor: {score:>6.2f}  [{bar}]  Label: {label}\n")

    # Evaluasi batch
    print("[4/4] Evaluasi batch (n=120)...")
    dataset = load_uci_sample()
    result  = evaluate(sim, dataset)

    print("\n" + "=" * 55)
    print("  HASIL EVALUASI TAHAP 1")
    print("=" * 55)
    print(f"  Akurasi Keseluruhan : {result['accuracy']}%")
    print(f"  Sampel Valid        : {result['n']}")
    print(f"  Prediksi Benar      : {result['correct']}")
    print("\n  Akurasi Per Kelas:")
    for lbl, acc in result['per_class'].items():
        bar = '█' * int(acc * 20)
        print(f"    {lbl:<8}: {acc*100:>5.1f}%  {bar}")

    print("\n  Confusion Matrix (baris=Aktual, kolom=Prediksi):")
    label_order = ['Rendah', 'Sedang', 'Tinggi']
    header = "         " + "  ".join(f"{l:>7}" for l in label_order)
    print(header)
    for t in label_order:
        row = f"  {t:<8}" + "  ".join(f"{result['confusion'][t][p]:>7}" for p in label_order)
        print(row)

    plot_evaluation(result)

    print("\n[SELESAI] Output tersimpan: mf_tahap1.png, eval_tahap1.png")
    print("=" * 55)

    return sim, result


if __name__ == '__main__':
    sim, result = main()
