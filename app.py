"""
=============================================================
  Aplikasi Streamlit — Prediksi Risiko Dropout Mahasiswa
  Tahap 1: Manual FIS (Mamdani)
  Jalankan: streamlit run app.py
=============================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Import FIS dari file utama
from fis_manual import build_fis, build_rules, predict, load_uci_sample, evaluate
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ─── Config ───
st.set_page_config(
    page_title="Prediksi Risiko Dropout",
    page_icon="🎓",
    layout="wide"
)

# ─── Cache FIS ───
@st.cache_resource
def get_fis():
    ipk, kehadiran, mk_gagal, status_ekon, risiko = build_fis()
    rules   = build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko)
    fis_ctrl = ctrl.ControlSystem(rules)
    sim      = ctrl.ControlSystemSimulation(fis_ctrl)
    return sim, ipk, kehadiran, mk_gagal, status_ekon, risiko

sim, ipk_var, kehadiran_var, mk_gagal_var, status_ekon_var, risiko_var = get_fis()

# ─── Header ───
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.caption("Tahap 1 — Manual Fuzzy Inference System (Mamdani) | UTS Soft Computing 2025/2026")
st.divider()

# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi", "📊 Membership Functions", "📈 Evaluasi Batch"])

# ════════════════════════════════════════
#  TAB 1 — PREDIKSI SINGLE INSTANCE
# ════════════════════════════════════════
with tab1:
    st.subheader("Input Data Mahasiswa")
    col1, col2 = st.columns(2)

    with col1:
        ipk_val  = st.slider("📚 IPK Semester",          0.0, 4.0, 2.5, 0.05,
                              help="Indeks Prestasi Kumulatif semester terakhir")
        had_val  = st.slider("🏫 Tingkat Kehadiran (%)", 0,   100, 70,  1,
                              help="Persentase kehadiran dalam satu semester")
    with col2:
        mkf_val  = st.slider("❌ Jumlah MK Gagal",       0,   10,  2,   1,
                              help="Jumlah mata kuliah dengan nilai E/gagal")
        eko_val  = st.slider("💰 Status Ekonomi",         0.0, 1.0, 0.5, 0.05,
                              help="0 = Sangat Rentan, 1 = Sangat Stabil")

    st.divider()

    if st.button("🚀 Hitung Risiko Dropout", type="primary", use_container_width=True):
        try:
            score, label = predict(sim, ipk_val, had_val, mkf_val, eko_val)

            # Warna berdasarkan label
            color_map = {'Rendah': '🟢', 'Sedang': '🟡', 'Tinggi': '🔴'}
            bg_map    = {'Rendah': '#d4edda', 'Sedang': '#fff3cd', 'Tinggi': '#f8d7da'}

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Skor Risiko",    f"{score:.1f} / 100")
            col_b.metric("Kategori",       f"{color_map[label]} {label}")
            col_c.metric("Kepercayaan",    f"{'Tinggi' if abs(score-50)>20 else 'Sedang'}")

            # Progress bar
            st.markdown(f"**Skor:** {score:.1f}")
            st.progress(int(score))

            # Interpretasi
            interp = {
                'Rendah': "✅ Mahasiswa menunjukkan performa akademik yang baik. Risiko dropout rendah.",
                'Sedang': "⚠️ Ada beberapa indikator yang perlu diperhatikan. Disarankan konseling akademik.",
                'Tinggi': "🚨 Mahasiswa berisiko tinggi dropout. Segera lakukan intervensi dosen wali."
            }
            st.info(interp[label])

        except Exception as e:
            st.error(f"Error inferensi: {e}. Coba ubah nilai input.")

# ════════════════════════════════════════
#  TAB 2 — MEMBERSHIP FUNCTIONS
# ════════════════════════════════════════
with tab2:
    st.subheader("Visualisasi Membership Functions — Tahap 1 (Manual)")

    COLORS = {
        'Rendah': '#1D9E75', 'Sedang': '#EF9F27', 'Tinggi': '#E24B4A',
        'Jarang': '#E24B4A', 'Cukup':  '#EF9F27', 'Rajin':  '#1D9E75',
        'Sedikit':'#1D9E75', 'Banyak': '#E24B4A',
        'Rentan': '#E24B4A', 'Stabil': '#1D9E75',
    }

    vars_info = [
        (ipk_var,         'IPK Semester',          'Nilai IPK'),
        (kehadiran_var,   'Tingkat Kehadiran',      'Kehadiran (%)'),
        (mk_gagal_var,    'Jumlah MK Gagal',        'Jumlah MK'),
        (status_ekon_var, 'Status Ekonomi',         'Encoded'),
        (risiko_var,      'Risiko Dropout (Output)','Skor Risiko'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Membership Functions — Tahap 1 (Manual FIS)', fontsize=13, fontweight='bold')
    axes_flat = axes.flatten()

    for ax, (var, title, xlabel) in zip(axes_flat, vars_info):
        for label in var.terms:
            mf_vals = var[label].mf
            color   = COLORS.get(label, '#888')
            ax.plot(var.universe, mf_vals, label=label, color=color, linewidth=2)
            ax.fill_between(var.universe, mf_vals, alpha=0.08, color=color)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('μ', fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    axes_flat[5].axis('off')
    axes_flat[5].text(0.5, 0.5,
        "FIS: Mamdani\nAND: Minimum\nOR: Maximum\nDefuzz: Centroid\nRules: 30",
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#eef2ff', edgecolor='#aab'))

    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════
#  TAB 3 — EVALUASI BATCH
# ════════════════════════════════════════
with tab3:
    st.subheader("Evaluasi Batch — Dataset Simulasi UCI (n=120)")

    if st.button("▶ Jalankan Evaluasi", type="primary"):
        with st.spinner("Menjalankan inferensi pada 120 sampel..."):
            dataset = load_uci_sample()
            result  = evaluate(sim, dataset)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi Keseluruhan", f"{result['accuracy']}%")
        col2.metric("Benar / Total",       f"{result['correct']} / {result['n']}")
        col3.metric("Akurasi Rendah",      f"{result['per_class']['Rendah']*100:.1f}%")
        col4.metric("Akurasi Tinggi",      f"{result['per_class']['Tinggi']*100:.1f}%")

        # Confusion matrix
        label_order = ['Rendah', 'Sedang', 'Tinggi']
        cm_array = np.array([[result['confusion'][t][p] for p in label_order]
                              for t in label_order])

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

        im = axes2[0].imshow(cm_array, cmap='Blues', vmin=0)
        axes2[0].set_xticks(range(3)); axes2[0].set_xticklabels(label_order)
        axes2[0].set_yticks(range(3)); axes2[0].set_yticklabels(label_order)
        axes2[0].set_xlabel('Prediksi', fontweight='bold')
        axes2[0].set_ylabel('Aktual',   fontweight='bold')
        axes2[0].set_title('Confusion Matrix')
        for i in range(3):
            for j in range(3):
                v = cm_array[i, j]
                c = 'white' if v > cm_array.max() * 0.6 else 'black'
                axes2[0].text(j, i, str(v), ha='center', va='center',
                              fontsize=14, fontweight='bold', color=c)
        plt.colorbar(im, ax=axes2[0])

        colors = ['#1D9E75', '#EF9F27', '#E24B4A']
        bars = axes2[1].bar(label_order,
                            [result['per_class'][l]*100 for l in label_order],
                            color=colors, width=0.5)
        axes2[1].set_ylim(0, 115)
        axes2[1].set_title('Akurasi Per Kelas')
        axes2[1].set_xlabel('Kelas Risiko', fontweight='bold')
        axes2[1].set_ylabel('Akurasi (%)',  fontweight='bold')
        axes2[1].axhline(result['accuracy'], color='gray', linestyle='--',
                         label=f"Overall: {result['accuracy']}%")
        axes2[1].legend()
        axes2[1].grid(axis='y', alpha=0.3)
        axes2[1].spines[['top', 'right']].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            axes2[1].text(bar.get_x() + bar.get_width()/2, h + 2,
                          f"{h:.1f}%", ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)
