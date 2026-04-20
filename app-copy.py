"""
=============================================================
  Prediksi Risiko Dropout Mahasiswa
  Sistem Terintegrasi: Manual FIS + Neuro-Fuzzy + GA Tuning
  UTS Soft Computing 2025/2026
=============================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Backend imports ──────────────────────────────────────────
from fis_manual import (build_fis, build_rules, predict,
                         load_uci_dataset, evaluate)
from fis_ann    import (NeuroFuzzyNet, prepare_dataset,
                         train_ann, predict_ann, evaluate_ann,
                         get_rule_weights)
from fis_ga     import (run_ga_tuning, build_fis_from_chromosome,
                         get_ga_mf_params)
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Shared dataset configuration (aligned with fis_manual.py)
DATASET_SAMPLE_N = 200
DATASET_RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Risiko Dropout",
    page_icon="🎓",
    layout="wide",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL COLOUR PALETTE  (shared by every plot)
# ════════════════════════════════════════════════════════════
C = {
    'Rendah': '#1D9E75', 'Sedang': '#EF9F27', 'Tinggi': '#E24B4A',
    'Jarang': '#E24B4A', 'Cukup':  '#EF9F27', 'Rajin':  '#1D9E75',
    'Sedikit':'#1D9E75', 'Banyak': '#E24B4A',
    'Rentan': '#E24B4A', 'Stabil': '#1D9E75',
    # per-method line styles for comparison plots
    'manual': '#3B82F6',   # blue
    'ga':     '#F59E0B',   # amber
    'ann':    '#8B5CF6',   # violet
}

# ════════════════════════════════════════════════════════════
#  CACHED RESOURCES  — built once per session
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Membangun Manual FIS…")
def get_manual_fis():
    """Build and return the baseline Manual Mamdani FIS."""
    ipk, kehadiran, mk_gagal, status_ekon, risiko = build_fis()
    rules    = build_rules(ipk, kehadiran, mk_gagal, status_ekon, risiko)
    fis_ctrl = ctrl.ControlSystem(rules)
    sim      = ctrl.ControlSystemSimulation(fis_ctrl)
    return sim, ipk, kehadiran, mk_gagal, status_ekon, risiko


# ── Retrieve cached Manual FIS ───────────────────────────────
sim_manual, ipk_var, kehadiran_var, mk_gagal_var, status_ekon_var, risiko_var = get_manual_fis()

# ── Session-state keys for trained models ───────────────────
if 'ann_model'       not in st.session_state: st.session_state['ann_model']       = None
if 'ann_loss'        not in st.session_state: st.session_state['ann_loss']        = None
if 'ann_acc'         not in st.session_state: st.session_state['ann_acc']         = None
if 'ga_best_sol'     not in st.session_state: st.session_state['ga_best_sol']     = None
if 'ga_best_fit'     not in st.session_state: st.session_state['ga_best_fit']     = None
if 'ga_fit_history'  not in st.session_state: st.session_state['ga_fit_history']  = None
if 'ga_pop_history'  not in st.session_state: st.session_state['ga_pop_history']  = None
if 'ga_fis_vars'     not in st.session_state: st.session_state['ga_fis_vars']     = None

# ════════════════════════════════════════════════════════════
#  HELPER — draw one confusion-matrix axes
# ════════════════════════════════════════════════════════════
def _draw_cm(ax, confusion, title, label_order=('Rendah', 'Sedang', 'Tinggi')):
    label_order = list(label_order)
    cm = np.array([[confusion[t][p] for p in label_order] for t in label_order])
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(label_order, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(label_order, fontsize=8)
    ax.set_xlabel('Prediksi', fontsize=8, fontweight='bold')
    ax.set_ylabel('Aktual',   fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold')
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            ax.text(j, i, str(v), ha='center', va='center', fontsize=11,
                    fontweight='bold',
                    color='white' if v > cm.max() * 0.55 else 'black')
    return im


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.caption("Sistem Pakar Terintegrasi: Manual FIS · Neuro-Fuzzy ANN · GA Tuning — UTS Soft Computing 2025/2026")
st.divider()

# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Prediksi",
    "📊 Membership Functions",
    "📈 Evaluasi & Perbandingan",
    "🧠 Tahap 2: Neuro-Fuzzy",
    "🧬 Tahap 3: GA Tuning",
    "🔬 Perbandingan MF",
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — PREDIKSI SINGLE INSTANCE (semua tiga metode)
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Input Data Mahasiswa")
    st.caption("Hasil prediksi ditampilkan dari ketiga metode secara bersamaan.")

    col1, col2 = st.columns(2)
    with col1:
        ipk_val = st.slider("📚 IPK Semester",          0.0, 4.0, 2.5, 0.05,
                             help="Indeks Prestasi Kumulatif semester terakhir")
        had_val = st.slider("🏫 Tingkat Kehadiran (%)", 0,   100, 70,  1,
                             help="Persentase kehadiran dalam satu semester")
    with col2:
        mkf_val = st.slider("❌ Jumlah MK Gagal",       0,   10,  2,   1,
                             help="Jumlah mata kuliah dengan nilai E/gagal")
        eko_val = st.slider("💰 Status Ekonomi",         0.0, 1.0, 0.5, 0.05,
                             help="0 = Sangat Rentan · 1 = Sangat Stabil")
    st.divider()

    if st.button("🚀 Hitung Risiko Dropout", type="primary", use_container_width=True):

        ICON = {'Rendah': '🟢', 'Sedang': '🟡', 'Tinggi': '🔴'}
        INTERP = {
            'Rendah': "✅ Performa akademik baik. Risiko dropout rendah.",
            'Sedang': "⚠️ Ada indikator yang perlu diperhatikan. Disarankan konseling akademik.",
            'Tinggi': "🚨 Risiko dropout tinggi. Segera lakukan intervensi dosen wali.",
        }

        # ── Manual FIS ──────────────────────────────────────
        try:
            score_m, label_m = predict(sim_manual, ipk_val, had_val, mkf_val, eko_val)
        except Exception as e:
            score_m, label_m = None, None
            st.warning(f"Manual FIS error: {e}")

        # ── GA-tuned FIS ─────────────────────────────────────
        score_g, label_g = None, None
        if st.session_state['ga_best_sol'] is not None:
            try:
                ga_result = build_fis_from_chromosome(st.session_state['ga_best_sol'])
                if ga_result:
                    sim_ga, *_ = ga_result
                    sim_ga.input['ipk']         = float(ipk_val)
                    sim_ga.input['kehadiran']   = float(had_val)
                    sim_ga.input['mk_gagal']    = float(mkf_val)
                    sim_ga.input['status_ekon'] = float(eko_val)
                    sim_ga.compute()
                    score_g = round(sim_ga.output['risiko'], 2)
                    label_g = 'Rendah' if score_g < 40 else ('Sedang' if score_g < 65 else 'Tinggi')
            except Exception as e:
                st.warning(f"GA FIS error: {e}")

        # ── ANN ──────────────────────────────────────────────
        score_a, label_a = None, None
        if st.session_state['ann_model'] is not None:
            try:
                score_a, label_a = predict_ann(
                    st.session_state['ann_model'],
                    ipk_val, had_val, mkf_val, eko_val
                )
            except Exception as e:
                st.warning(f"ANN error: {e}")

        # ── Display side-by-side ─────────────────────────────
        col_m, col_g, col_a_disp = st.columns(3)

        with col_m:
            st.markdown("#### 🔵 Manual FIS")
            if score_m is not None:
                st.metric("Skor",     f"{score_m:.1f} / 100")
                st.metric("Kategori", f"{ICON[label_m]} {label_m}")
                st.progress(int(score_m))
                st.info(INTERP[label_m])
            else:
                st.error("Gagal menghitung")

        with col_g:
            st.markdown("#### 🟡 GA-Tuned FIS")
            if score_g is not None:
                st.metric("Skor",     f"{score_g:.1f} / 100")
                st.metric("Kategori", f"{ICON[label_g]} {label_g}")
                st.progress(int(score_g))
                st.info(INTERP[label_g])
                delta = score_g - score_m if score_m is not None else 0
                st.caption(f"Δ vs Manual: {delta:+.1f} poin")
            else:
                st.info("Jalankan GA Tuning di Tab 5 terlebih dahulu.")

        with col_a_disp:
            st.markdown("#### 🟣 Neuro-Fuzzy ANN")
            if score_a is not None:
                st.metric("Skor",     f"{score_a:.1f} / 100")
                st.metric("Kategori", f"{ICON[label_a]} {label_a}")
                st.progress(int(score_a))
                st.info(INTERP[label_a])
                delta = score_a - score_m if score_m is not None else 0
                st.caption(f"Δ vs Manual: {delta:+.1f} poin")
            else:
                st.info("Latih ANN di Tab 4 terlebih dahulu.")


# ════════════════════════════════════════════════════════════
#  TAB 2 — MEMBERSHIP FUNCTIONS (Manual FIS)
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Membership Functions — Manual FIS (Mamdani)")
    st.caption("Fungsi keanggotaan yang didefinisikan secara manual oleh pakar.")

    vars_info = [
        (ipk_var,         'IPK Semester',           'Nilai IPK'),
        (kehadiran_var,   'Tingkat Kehadiran',       'Kehadiran (%)'),
        (mk_gagal_var,    'Jumlah MK Gagal',         'Jumlah MK'),
        (status_ekon_var, 'Status Ekonomi',          'Encoded (0=Rentan, 1=Stabil)'),
        (risiko_var,      'Risiko Dropout (Output)', 'Skor Risiko (0–100)'),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle("Membership Functions — Manual FIS", fontsize=13, fontweight='bold')
    axes2_flat = axes2.flatten()

    for ax, (var, title, xlabel) in zip(axes2_flat, vars_info):
        for lbl in var.terms:
            mf = var[lbl].mf
            col = C.get(lbl, '#888')
            ax.plot(var.universe, mf, label=lbl, color=col, linewidth=2)
            ax.fill_between(var.universe, mf, alpha=0.10, color=col)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('μ', fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    axes2_flat[5].axis('off')
    axes2_flat[5].text(
        0.5, 0.5,
        "FIS Type   : Mamdani\n"
        "AND op     : Minimum\n"
        "OR op      : Maximum\n"
        "Defuzz     : Centroid\n"
        "Total Rules: 31",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4ff',
                  edgecolor='#aabbdd', linewidth=1.2),
        transform=axes2_flat[5].transAxes,
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ════════════════════════════════════════════════════════════
#  TAB 3 — EVALUASI & PERBANDINGAN BATCH
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Evaluasi Batch — Perbandingan Ketiga Metode (UCI #697)")
    st.caption(
        f"Klik tombol di bawah untuk menjalankan evaluasi pada dataset UCI #697 "
        f"(stratified sample: {DATASET_SAMPLE_N} per kelas). "
        "GA dan ANN hanya ditampilkan jika sudah dilatih di Tab 4 / Tab 5."
    )

    if st.button("▶ Jalankan Evaluasi Batch", type="primary"):
        dataset      = load_uci_dataset(
            sample_n=DATASET_SAMPLE_N,
            random_state=DATASET_RANDOM_STATE
        )
        label_order  = ['Rendah', 'Sedang', 'Tinggi']

        # ── Manual ──────────────────────────────────────────
        with st.spinner("Evaluasi Manual FIS…"):
            res_m = evaluate(sim_manual, dataset)

        # ── GA ──────────────────────────────────────────────
        res_g = None
        if st.session_state['ga_best_sol'] is not None:
            with st.spinner("Evaluasi GA-Tuned FIS…"):
                ga_result = build_fis_from_chromosome(st.session_state['ga_best_sol'])
                if ga_result:
                    sim_ga2, *_ = ga_result
                    res_g = evaluate(sim_ga2, dataset)

        # ── ANN ─────────────────────────────────────────────
        res_a = None
        if st.session_state['ann_model'] is not None:
            with st.spinner("Evaluasi ANN…"):
                res_a = evaluate_ann(st.session_state['ann_model'], dataset)

        # ── Accuracy summary row ─────────────────────────────
        st.markdown("### Akurasi Keseluruhan")
        n_cols  = 1 + (1 if res_g else 0) + (1 if res_a else 0)
        acc_cols = st.columns(n_cols)
        acc_cols[0].metric("🔵 Manual FIS",
                           f"{res_m['accuracy']}%",
                           help=f"Valid: {res_m['n']} | Skipped: {res_m['skipped']}")
        idx = 1
        if res_g:
            delta_g = round(res_g['accuracy'] - res_m['accuracy'], 2)
            acc_cols[idx].metric("🟡 GA-Tuned FIS",
                                 f"{res_g['accuracy']}%",
                                 delta=f"{delta_g:+.2f}%")
            idx += 1
        if res_a:
            delta_a = round(res_a['accuracy'] - res_m['accuracy'], 2)
            acc_cols[idx].metric("🟣 Neuro-Fuzzy ANN",
                                 f"{res_a['accuracy']}%",
                                 delta=f"{delta_a:+.2f}%")

        st.divider()

        # ── Confusion matrices ───────────────────────────────
        st.markdown("### Confusion Matrices")
        cm_count = 1 + (1 if res_g else 0) + (1 if res_a else 0)
        fig3, axes3 = plt.subplots(1, cm_count, figsize=(5 * cm_count, 4.5))
        if cm_count == 1:
            axes3 = [axes3]
        _draw_cm(axes3[0], res_m['confusion'], "Manual FIS")
        ci = 1
        if res_g:
            _draw_cm(axes3[ci], res_g['confusion'], "GA-Tuned FIS"); ci += 1
        if res_a:
            _draw_cm(axes3[ci], res_a['confusion'], "Neuro-Fuzzy ANN")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.divider()

        # ── Per-class accuracy bars ──────────────────────────
        st.markdown("### Akurasi Per Kelas")
        bar_colors = [C['Rendah'], C['Sedang'], C['Tinggi']]
        fig3b, axes3b = plt.subplots(1, cm_count, figsize=(5 * cm_count, 4),
                                     sharey=True)
        if cm_count == 1:
            axes3b = [axes3b]

        def _draw_per_class(ax, res, title, color_title):
            vals = [res['per_class'][l] * 100 for l in label_order]
            bars = ax.bar(label_order, vals, color=bar_colors,
                          width=0.5, edgecolor='white', linewidth=1.2)
            ax.axhline(res['accuracy'], color='grey', linestyle='--',
                       linewidth=1, label=f"Overall {res['accuracy']}%")
            ax.set_ylim(0, 115)
            ax.set_title(f"{color_title} {title}", fontsize=9, fontweight='bold')
            ax.set_ylabel("Akurasi (%)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 2, f"{v:.0f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        _draw_per_class(axes3b[0], res_m, "Manual FIS", "🔵")
        ci = 1
        if res_g:
            _draw_per_class(axes3b[ci], res_g, "GA-Tuned FIS", "🟡"); ci += 1
        if res_a:
            _draw_per_class(axes3b[ci], res_a, "Neuro-Fuzzy ANN", "🟣")
        plt.tight_layout()
        st.pyplot(fig3b)
        plt.close(fig3b)


# ════════════════════════════════════════════════════════════
#  TAB 4 — TAHAP 2: NEURO-FUZZY ANN
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧠 Neuro-Fuzzy (ANN) — Tuning Rule Base via Backpropagation")

    with st.expander("📖 Cara Kerja", expanded=False):
        st.markdown(
            """
            **Bagaimana ANN 'men-tune' FIS?**

            1. **Target belajar** — ANN dilatih untuk mereproduksi *skor output* dari Manual FIS
               pada dataset UCI #697 (stratified sample; regression, bukan klasifikasi).
            2. **Arsitektur merefleksikan FIS** —
               - *Layer 1 (12 nodes)* mewakili fungsi keanggotaan (3 MF × 4 variabel input).
               - *Layer 2 (30 nodes)* mewakili 30 rule yang sama persis dengan Manual FIS.
               - *Layer 3 (1 node)* adalah skor risiko akhir (0–100).
            3. **Tuning rule** — Setelah training, **bobot Layer 2** mencerminkan seberapa
               penting setiap rule. Rule dengan bobot tinggi lebih berpengaruh terhadap output
               daripada yang tertulis secara manual — inilah "tuning" yang dilakukan ANN.
            4. **Output** — Model yang terlatih digunakan langsung untuk prediksi baru,
               menggantikan inferensi skfuzzy dengan forward-pass neural network.
            """
        )

    st.divider()
    col_a1, col_a2 = st.columns([1, 2])

    with col_a1:
        st.markdown("#### Konfigurasi Training")
        ann_epochs = st.slider("Jumlah Epochs", 50, 500, 200, 25,
                               help="Lebih banyak epoch = fitting lebih baik, tapi lebih lama")
        ann_lr     = st.select_slider("Learning Rate",
                                      options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                      value=0.001)
        ann_bs     = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)

        st.markdown("#### Arsitektur")
        st.code(
            "Input (4)  →  [Fuzzifikasi]\n"
            "           →  Linear(4→12) + Sigmoid\n"
            "           →  [Rule Base]\n"
            "           →  Linear(12→30) + ReLU\n"
            "           →  [Defuzzifikasi]\n"
            "           →  Linear(30→1) + Sigmoid×100\n"
            "Output: Skor Risiko [0–100]",
            language="text",
        )

        run_ann = st.button("🔥 Latih ANN Sekarang", type="primary",
                            use_container_width=True)

    with col_a2:
        if run_ann:
            dataset  = load_uci_dataset(
                sample_n=DATASET_SAMPLE_N,
                random_state=DATASET_RANDOM_STATE
            )
            prog_bar = st.progress(0)
            stat_txt = st.empty()

            # ── Real training callback ───────────────────────
            def _ann_cb(epoch, total, loss):
                prog_bar.progress(epoch / total)
                stat_txt.text(f"Epoch {epoch}/{total}  —  MSE Loss: {loss:.4f}")

            with st.spinner("Training Neuro-Fuzzy ANN…"):
                model_trained, loss_hist, acc_hist = train_ann(
                    sim_manual, dataset,
                    epochs=ann_epochs,
                    lr=ann_lr,
                    batch_size=ann_bs,
                    progress_callback=_ann_cb,
                )

            st.session_state['ann_model'] = model_trained
            st.session_state['ann_loss']  = loss_hist
            st.session_state['ann_acc']   = acc_hist
            prog_bar.progress(1.0)
            stat_txt.text(f"Training selesai! MSE akhir: {loss_hist[-1]:.4f}")
            st.success(f"✅ Training selesai — {ann_epochs} epochs | "
                       f"Final MSE: {loss_hist[-1]:.4f} | "
                       f"Final Acc: {acc_hist[-1]*100:.1f}%")

        # ── Show training curves if model exists ─────────────
        if st.session_state['ann_loss'] is not None:
            loss_h = st.session_state['ann_loss']
            acc_h  = st.session_state['ann_acc']
            epochs_run = len(loss_h)

            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4))
            fig4.suptitle("Kurva Training Neuro-Fuzzy ANN", fontsize=11, fontweight='bold')

            # Loss curve
            ax4a.plot(range(1, epochs_run + 1), loss_h,
                      color=C['ann'], linewidth=1.8)
            ax4a.set_xlabel("Epoch"); ax4a.set_ylabel("MSE Loss")
            ax4a.set_title("Loss (MSE) per Epoch")
            ax4a.grid(True, alpha=0.3)
            ax4a.spines[['top', 'right']].set_visible(False)

            # Accuracy curve
            ax4b.plot(range(1, epochs_run + 1), [a * 100 for a in acc_h],
                      color=C['Rendah'], linewidth=1.8)
            ax4b.set_xlabel("Epoch"); ax4b.set_ylabel("Akurasi (%)")
            ax4b.set_title("Akurasi Klasifikasi per Epoch")
            ax4b.set_ylim(0, 105)
            ax4b.grid(True, alpha=0.3)
            ax4b.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            # ── Rule importance bar chart ────────────────────
            st.markdown("#### Kepentingan Relatif Setiap Rule (Bobot Layer 2)")
            st.caption(
                "Tinggi rendahnya bar merepresentasikan seberapa besar kontribusi "
                "setiap rule terhadap output akhir setelah training. "
                "Ini adalah bentuk 'tuning' yang dilakukan ANN terhadap rule base manual."
            )
            rule_w = get_rule_weights(st.session_state['ann_model'])
            rule_labels = [f"R{i+1}" for i in range(30)]

            # Colour-code: R1–10 Tinggi, R11–22 Sedang, R23–30 Rendah
            bar_cols = (
                [C['Tinggi']] * 10 +
                [C['Sedang']] * 12 +
                [C['Rendah']] * 8
            )
            fig4c, ax4c = plt.subplots(figsize=(13, 3.5))
            bars = ax4c.bar(rule_labels, rule_w, color=bar_cols,
                            edgecolor='white', linewidth=0.8)
            ax4c.set_xlabel("Rule (R1–10: Tinggi | R11–22: Sedang | R23–30: Rendah)",
                            fontsize=9)
            ax4c.set_ylabel("Normalised Importance", fontsize=9)
            ax4c.set_title("ANN Rule Weight Importance (after training)", fontsize=10,
                           fontweight='bold')
            ax4c.set_ylim(0, 1.15)
            ax4c.axhline(np.mean(rule_w), color='grey', linestyle='--',
                         linewidth=1, label=f"Mean: {np.mean(rule_w):.2f}")
            ax4c.legend(fontsize=8)
            ax4c.grid(axis='y', alpha=0.3)
            ax4c.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4c)
            plt.close(fig4c)

            # ── Top-5 most & least important rules ───────────
            top5    = np.argsort(rule_w)[::-1][:5]
            bottom5 = np.argsort(rule_w)[:5]
            rule_names = (
                [f"R{i+1} (Tinggi)" for i in range(10)] +
                [f"R{i+1} (Sedang)" for i in range(10, 22)] +
                [f"R{i+1} (Rendah)" for i in range(22, 30)]
            )
            c_top, c_bot = st.columns(2)
            with c_top:
                st.markdown("**Top-5 Rule Paling Penting:**")
                for i in top5:
                    st.markdown(f"- {rule_names[i]} — bobot `{rule_w[i]:.3f}`")
            with c_bot:
                st.markdown("**Top-5 Rule Paling Lemah:**")
                for i in bottom5:
                    st.markdown(f"- {rule_names[i]} — bobot `{rule_w[i]:.3f}`")

        else:
            st.info("Klik **Latih ANN Sekarang** untuk memulai training.")


# ════════════════════════════════════════════════════════════
#  TAB 5 — TAHAP 3: GENETIC ALGORITHM TUNING
# ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧬 GA Tuning — Optimasi Titik Koordinat Membership Functions")

    with st.expander("📖 Cara Kerja", expanded=False):
        st.markdown(
            """
            **Bagaimana GA 'men-tune' FIS?**

            1. **Kromosom (20 gen)** — Setiap individu mewakili satu set titik koordinat
               batas/puncak dari seluruh MF pada 5 variabel FIS (IPK, Kehadiran, MK Gagal,
               Status Ekonomi, dan Risiko output).
            2. **Fungsi Fitness** — Setiap kromosom di-*decode* menjadi FIS skfuzzy lengkap,
               lalu dievaluasi akurasinya pada dataset UCI #697 (stratified sample).
               Fitness = akurasi klasifikasi.
            3. **Seleksi & Evolusi** — GA memilih kromosom terbaik, melakukan crossover dan
               mutasi, menghasilkan generasi baru yang (rata-rata) lebih baik.
            4. **Output** — Kromosom terbaik digunakan untuk membangun FIS GA-Tuned yang
               titik koordinat MF-nya berbeda dari manual, tetapi dengan topologi rule yang sama.
            """
        )

    st.divider()
    col_g1, col_g2 = st.columns([1, 2])

    with col_g1:
        st.markdown("#### Konfigurasi GA")
        ga_pop  = st.slider("Population Size", 10, 100, 30, 5)
        ga_gen  = st.slider("Generations",     5,  50,  15, 5)

        st.markdown("#### Kromosom (20 Gen)")
        st.caption(
            "Setiap gen adalah titik koordinat MF yang dioptimasi. "
            "Batas nilai setiap gen disesuaikan dengan ranah FIS manual."
        )
        gene_info = {
            "IPK":        ["Rendah upper", "Sedang peak", "Tinggi lower", "Rendah inner"],
            "Kehadiran":  ["Jarang upper", "Cukup peak",  "Rajin lower",  "Jarang inner"],
            "MK Gagal":   ["Sedikit upper","Sedang peak", "Banyak lower", "Sedikit inner", "Banyak upper"],
            "Eko":        ["Rentan mean",  "Stabil mean"],
            "Risiko":     ["Rendah upper", "Sedang peak", "Tinggi lower", "Rendah lower",  "Tinggi upper"],
        }
        for var_name, gens in gene_info.items():
            st.markdown(f"**{var_name}:** {', '.join(gens)}")

        run_ga = st.button("🧬 Jalankan GA Optimization", type="primary",
                           use_container_width=True)

    with col_g2:
        if run_ga:
            gen_placeholder   = st.empty()
            prog_ga           = st.progress(0)
            live_fitness_data = []
            live_pop_data     = []

            def _ga_cb(ga_inst, gen_num, best_fit):
                live_fitness_data.append(best_fit)
                live_pop_data.append((
                    ga_inst.population.copy(),
                    ga_inst.last_generation_fitness.copy()
                ))
                prog_ga.progress(gen_num / ga_gen)
                gen_placeholder.text(
                    f"Generation {gen_num}/{ga_gen}  —  Best Fitness: {best_fit:.4f}"
                )

            with st.spinner("GA sedang berjalan…"):
                best_sol, best_fit, fit_hist, pop_hist = run_ga_tuning(
                    pop_size=ga_pop,
                    num_gen=ga_gen,
                    on_generation=_ga_cb,
                    dataset_sample_n=DATASET_SAMPLE_N,
                    dataset_random_state=DATASET_RANDOM_STATE,
                )

            st.session_state['ga_best_sol']    = best_sol
            st.session_state['ga_best_fit']    = best_fit
            st.session_state['ga_fit_history'] = fit_hist
            st.session_state['ga_pop_history'] = pop_hist

            # Build and cache GA FIS vars for comparison tab
            ga_build = build_fis_from_chromosome(best_sol)
            if ga_build:
                _, *ga_vars = ga_build
                st.session_state['ga_fis_vars'] = ga_vars

            prog_ga.progress(1.0)
            gen_placeholder.text(
                f"✅ Selesai! Best Fitness (Akurasi): {best_fit*100:.2f}%"
            )
            st.success(
                f"GA Optimization selesai — {ga_gen} generasi, "
                f"populasi {ga_pop} — Best Accuracy: **{best_fit*100:.2f}%**"
            )

        # ── Visualisasi GA (ditampilkan jika sudah ada data) ──
        if st.session_state['ga_fit_history'] is not None:
            fit_hist = st.session_state['ga_fit_history']
            pop_hist = st.session_state['ga_pop_history']
            n_gens   = len(fit_hist)

            # ── Plot 1: Fitness curve ────────────────────────
            st.markdown("#### Kurva Fitness per Generasi")
            fig5a, ax5a = plt.subplots(figsize=(10, 3.5))

            best_per_gen = [float(np.max(pf[1])) for pf in pop_hist]
            worst_per_gen= [float(np.min(pf[1])) for pf in pop_hist]
            mean_per_gen = [float(np.mean(pf[1])) for pf in pop_hist]
            gens_x       = list(range(1, n_gens + 1))

            ax5a.fill_between(gens_x, worst_per_gen, best_per_gen,
                              alpha=0.15, color=C['ga'], label="Min–Max range")
            ax5a.plot(gens_x, best_per_gen,  color=C['ga'],     linewidth=2.2,
                      label="Best fitness", marker='o', markersize=4)
            ax5a.plot(gens_x, mean_per_gen,  color='#6B7280',   linewidth=1.4,
                      linestyle='--', label="Mean fitness")
            ax5a.plot(gens_x, worst_per_gen, color=C['Tinggi'], linewidth=1.2,
                      linestyle=':', label="Worst fitness")

            ax5a.set_xlabel("Generasi", fontsize=9)
            ax5a.set_ylabel("Fitness (Akurasi)", fontsize=9)
            ax5a.set_title("Evolusi Fitness GA per Generasi", fontsize=10, fontweight='bold')
            ax5a.legend(fontsize=8)
            ax5a.grid(True, alpha=0.3)
            ax5a.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5a)
            plt.close(fig5a)

            st.divider()

            # ── Plot 2: Population scatter per generation ────
            st.markdown("#### Distribusi Populasi per Generasi — Gen Terpilih vs Baseline")
            st.caption(
                "Setiap titik adalah satu individu (kromosom) dalam populasi generasi tersebut. "
                "Garis merah = nilai baseline Manual FIS (gen default). "
                "Titik hijau = individu terbaik generasi itu. "
                "Titik biru = populasi umum. "
                "Titik merah tua = individu terburuk generasi itu."
            )

            # Baseline chromosome (manual FIS parameters, flattened)
            BASELINE = np.array([
                2.2, 2.5, 2.8,    # genes 0–2: IPK
                60,  70,  85,     # genes 3–5: Kehadiran
                3,   4,   7,      # genes 6–8: MK Gagal
                0.0, 1.0,         # genes 9–10: StatusEkon
                35,  50,  75,     # genes 11–13: Risiko
                1.5, 40,  1.0,    # genes 14–16: inner points
                45,  75,  10,     # genes 17–19: crosses/uppers
            ])

            # Let user pick which generation and which gene pair to inspect
            sel_gen = st.slider("Pilih Generasi untuk Ditampilkan Detail",
                                1, n_gens, min(n_gens, 5))
            pop_sel, fit_sel = pop_hist[sel_gen - 1]

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                gene_x_idx = st.selectbox(
                    "Gen X (sumbu horizontal)",
                    options=list(range(20)),
                    format_func=lambda i: [
                        "IPK Rendah upper","IPK Sedang peak","IPK Tinggi lower",
                        "Had Jarang upper","Had Cukup peak","Had Rajin lower",
                        "MK Sedikit upper","MK Sedang peak","MK Banyak lower",
                        "Eko Rentan mean","Eko Stabil mean",
                        "Risiko Rendah upper","Risiko Sedang peak","Risiko Tinggi lower",
                        "IPK Rendah inner","Had Jarang inner","MK Sedikit inner",
                        "Risiko Rendah lower","Risiko Tinggi upper","MK Banyak upper",
                    ][i],
                    index=0,
                )
            with g_col2:
                gene_y_idx = st.selectbox(
                    "Gen Y (sumbu vertikal)",
                    options=list(range(20)),
                    format_func=lambda i: [
                        "IPK Rendah upper","IPK Sedang peak","IPK Tinggi lower",
                        "Had Jarang upper","Had Cukup peak","Had Rajin lower",
                        "MK Sedikit upper","MK Sedang peak","MK Banyak lower",
                        "Eko Rentan mean","Eko Stabil mean",
                        "Risiko Rendah upper","Risiko Sedang peak","Risiko Tinggi lower",
                        "IPK Rendah inner","Had Jarang inner","MK Sedikit inner",
                        "Risiko Rendah lower","Risiko Tinggi upper","MK Banyak upper",
                    ][i],
                    index=1,
                )

            # ── Figure: side-by-side all-gen + selected-gen scatter ──
            fig5b, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(13, 5))
            fig5b.suptitle(
                f"Distribusi Populasi — Gen {gene_x_idx} vs Gen {gene_y_idx}",
                fontsize=11, fontweight='bold'
            )

            # Left: all generations overlaid, colour by generation
            cmap_gen = plt.get_cmap('plasma', n_gens)
            for gi, (pop_g, fit_g) in enumerate(pop_hist):
                ax5l.scatter(
                    pop_g[:, gene_x_idx], pop_g[:, gene_y_idx],
                    color=cmap_gen(gi), alpha=0.45, s=18,
                    label=f"Gen {gi+1}" if gi in [0, n_gens // 2, n_gens - 1] else None
                )
            # Baseline marker
            ax5l.axvline(BASELINE[gene_x_idx], color='red', linestyle='--',
                         linewidth=1.2, alpha=0.7, label="Baseline X")
            ax5l.axhline(BASELINE[gene_y_idx], color='red', linestyle='--',
                         linewidth=1.2, alpha=0.7, label="Baseline Y")
            ax5l.scatter(BASELINE[gene_x_idx], BASELINE[gene_y_idx],
                         color='red', s=120, marker='*', zorder=5, label="Baseline")
            ax5l.set_xlabel(f"Gen {gene_x_idx}", fontsize=8)
            ax5l.set_ylabel(f"Gen {gene_y_idx}", fontsize=8)
            ax5l.set_title("Semua Generasi (warna = generasi)", fontsize=9)
            ax5l.legend(fontsize=7, loc='best')
            ax5l.grid(True, alpha=0.2)
            ax5l.spines[['top', 'right']].set_visible(False)

            # Right: single selected generation with extreme highlights
            best_idx  = int(np.argmax(fit_sel))
            worst_idx = int(np.argmin(fit_sel))

            # All individuals
            ax5r.scatter(pop_sel[:, gene_x_idx], pop_sel[:, gene_y_idx],
                         c=fit_sel, cmap='RdYlGn', vmin=0, vmax=1,
                         s=60, alpha=0.8, zorder=2, label="Populasi")

            # Best individual
            ax5r.scatter(pop_sel[best_idx, gene_x_idx],
                         pop_sel[best_idx, gene_y_idx],
                         color='#16A34A', s=200, marker='*', zorder=5,
                         label=f"Best (fit={fit_sel[best_idx]:.3f})")

            # Worst individual
            ax5r.scatter(pop_sel[worst_idx, gene_x_idx],
                         pop_sel[worst_idx, gene_y_idx],
                         color='#DC2626', s=120, marker='v', zorder=5,
                         label=f"Worst (fit={fit_sel[worst_idx]:.3f})")

            # Baseline
            ax5r.axvline(BASELINE[gene_x_idx], color='red', linestyle='--',
                         linewidth=1.2, alpha=0.6)
            ax5r.axhline(BASELINE[gene_y_idx], color='red', linestyle='--',
                         linewidth=1.2, alpha=0.6)
            ax5r.scatter(BASELINE[gene_x_idx], BASELINE[gene_y_idx],
                         color='red', s=100, marker='*', zorder=6, label="Baseline")

            sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                       norm=plt.Normalize(vmin=0, vmax=1))
            plt.colorbar(sm, ax=ax5r, label="Fitness")
            ax5r.set_xlabel(f"Gen {gene_x_idx}", fontsize=8)
            ax5r.set_ylabel(f"Gen {gene_y_idx}", fontsize=8)
            ax5r.set_title(f"Generasi {sel_gen} — Best / Worst / Baseline", fontsize=9)
            ax5r.legend(fontsize=7, loc='best')
            ax5r.grid(True, alpha=0.2)
            ax5r.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig5b)
            plt.close(fig5b)

            st.divider()

            # ── Plot 3: Fitness spread box-per-generation ────
            st.markdown("#### Box Plot Distribusi Fitness per Generasi")
            st.caption(
                "Setiap box menggambarkan distribusi fitness seluruh populasi dalam satu generasi. "
                "Semakin kotak menyempit dan naik, semakin konvergen dan membaik populasi."
            )
            fig5c, ax5c = plt.subplots(figsize=(max(8, n_gens * 0.6 + 2), 4))
            all_fits = [pop_hist[i][1] for i in range(n_gens)]
            bp = ax5c.boxplot(all_fits, patch_artist=True, notch=False,
                              positions=range(1, n_gens + 1))
            cmap_box = plt.get_cmap('YlGn', n_gens)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(cmap_box(i))
                patch.set_alpha(0.75)
            for element in ['whiskers', 'caps', 'medians', 'fliers']:
                for item in bp[element]:
                    item.set_color('#374151')

            ax5c.plot(range(1, n_gens + 1), best_per_gen,
                      color=C['ga'], linewidth=1.8, marker='o',
                      markersize=4, label="Best fitness")
            ax5c.set_xlabel("Generasi", fontsize=9)
            ax5c.set_ylabel("Fitness (Akurasi)", fontsize=9)
            ax5c.set_title("Distribusi Fitness Populasi per Generasi", fontsize=10,
                           fontweight='bold')
            ax5c.legend(fontsize=8)
            ax5c.grid(axis='y', alpha=0.3)
            ax5c.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5c)
            plt.close(fig5c)

            st.divider()

            # ── Best chromosome parameters ───────────────────
            st.markdown("#### Parameter Kromosom Terbaik")
            st.caption("Nilai titik koordinat MF yang ditemukan GA, dibandingkan dengan nilai manual (baseline).")
            mf_params = get_ga_mf_params(st.session_state['ga_best_sol'])
            gene_labels = [
                "IPK Rendah upper","IPK Sedang peak","IPK Tinggi lower",
                "Had Jarang upper","Had Cukup peak","Had Rajin lower",
                "MK Sedikit upper","MK Sedang peak","MK Banyak lower",
                "Eko Rentan mean","Eko Stabil mean",
                "Risiko Rendah upper","Risiko Sedang peak","Risiko Tinggi lower",
                "IPK Rendah inner","Had Jarang inner","MK Sedikit inner",
                "Risiko Rendah lower","Risiko Tinggi upper","MK Banyak upper",
            ]
            best_sol = st.session_state['ga_best_sol']
            rows = []
            for i, (lbl, bv, bl) in enumerate(zip(gene_labels, best_sol, BASELINE)):
                rows.append({
                    "Gen": i,
                    "Parameter": lbl,
                    "GA Tuned": f"{bv:.4f}",
                    "Baseline (Manual)": f"{bl:.4f}",
                    "Δ": f"{bv - bl:+.4f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Gen"), use_container_width=True)

        else:
            st.info("Klik **Jalankan GA Optimization** untuk memulai.")


# ════════════════════════════════════════════════════════════
#  TAB 6 — PERBANDINGAN MF (Manual vs GA vs ANN)
# ════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔬 Perbandingan Membership Functions — Manual vs GA vs ANN")
    st.caption(
        "Grafik ini menempatkan MF dari ketiga metode pada sumbu yang sama sehingga "
        "pergeseran titik koordinat akibat tuning GA dan efek bobot ANN dapat dilihat langsung."
    )

    # Check availability
    ga_ready  = st.session_state['ga_fis_vars'] is not None
    ann_ready = st.session_state['ann_model']   is not None

    if not ga_ready and not ann_ready:
        st.warning(
            "Latih **ANN** (Tab 4) dan/atau jalankan **GA** (Tab 5) terlebih dahulu "
            "agar perbandingan MF dapat ditampilkan."
        )
    else:
        # ── Retrieve GA MF variables ─────────────────────────
        if ga_ready:
            ga_ipk, ga_kehadiran, ga_mk, ga_eko, ga_risiko = (
                st.session_state['ga_fis_vars']
            )

        # ── ANN-implied MF shift ─────────────────────────────
        # The ANN doesn't have explicit MF curves, but we can derive an
        # "effective sensitivity" per input by looking at mf_layer weights.
        # We visualise this as a soft overlay (Gaussian centred on each MF peak,
        # scaled by the ANN's mean absolute weight for that MF node).
        if ann_ready:
            import torch
            model_ann = st.session_state['ann_model']
            with torch.no_grad():
                mf_w = model_ann.mf_layer.weight.abs().mean(dim=0).numpy()  # (12,)
            # mf_w[0:3]  → IPK MFs  (Rendah, Sedang, Tinggi)
            # mf_w[3:6]  → Kehadiran (Jarang, Cukup, Rajin)
            # mf_w[6:9]  → MK_Gagal  (Sedikit, Sedang, Banyak)
            # mf_w[9:12] → StatusEkon (Rentan, Stabil — only 2 but padded to 3)
            if mf_w.max() > 0:
                mf_w = mf_w / mf_w.max()

        # ── Subplot grid: 5 variables ─────────────────────────
        fig6, axes6 = plt.subplots(2, 3, figsize=(16, 9))
        fig6.suptitle(
            "Perbandingan Membership Functions\n"
            "Biru = Manual  |  Kuning = GA-Tuned  |  Ungu = ANN-weighted overlay",
            fontsize=11, fontweight='bold', y=1.01
        )
        axes6_flat = axes6.flatten()

        vars_manual  = [ipk_var, kehadiran_var, mk_gagal_var,
                        status_ekon_var, risiko_var]
        vars_ga      = ([ga_ipk, ga_kehadiran, ga_mk, ga_eko, ga_risiko]
                        if ga_ready else [None] * 5)
        # MF node index ranges per variable
        mf_node_ranges = [(0, 3), (3, 6), (6, 9), (9, 11), (9, 11)]
        titles = ["IPK Semester", "Tingkat Kehadiran",
                  "Jumlah MK Gagal", "Status Ekonomi", "Risiko Dropout (Output)"]
        xlabels= ["Nilai IPK", "Kehadiran (%)",
                  "Jumlah MK", "Encoded (0=Rentan)", "Skor Risiko (0–100)"]

        for idx, (ax, var_m, var_g, title, xlabel) in enumerate(
                zip(axes6_flat, vars_manual, vars_ga, titles, xlabels)):

            terms = list(var_m.terms.keys())

            # ── Manual MF (solid blue) ───────────────────────
            for ti, term in enumerate(terms):
                mf = var_m[term].mf
                col = C.get(term, '#888')
                ax.plot(var_m.universe, mf,
                        color=C['manual'], linewidth=2.0,
                        label=f"Manual {term}" if ti == 0 else "_")
                ax.fill_between(var_m.universe, mf,
                                alpha=0.07, color=C['manual'])

            # term-coloured label lines (thin, for legend readability)
            for term in terms:
                mf = var_m[term].mf
                ax.plot(var_m.universe, mf,
                        color=C.get(term, '#888'), linewidth=0.8,
                        linestyle='--', alpha=0.6)

            # ── GA-Tuned MF (dashed amber) ───────────────────
            if ga_ready and var_g is not None:
                for ti, term in enumerate(list(var_g.terms.keys())):
                    mf_g = var_g[term].mf
                    ax.plot(var_g.universe, mf_g,
                            color=C['ga'], linewidth=2.0, linestyle='--',
                            label=f"GA {term}" if ti == 0 else "_")
                    ax.fill_between(var_g.universe, mf_g,
                                    alpha=0.07, color=C['ga'])

            # ── ANN sensitivity overlay (violet shaded Gaussians) ──
            if ann_ready:
                node_start, node_end = mf_node_ranges[idx]
                uni = var_m.universe
                for ni, term in enumerate(terms):
                    if node_start + ni >= len(mf_w):
                        break
                    w      = mf_w[node_start + ni]
                    mf_ref = var_m[term].mf
                    peak   = float(uni[np.argmax(mf_ref)])
                    spread = (uni[-1] - uni[0]) * 0.12
                    gauss  = w * np.exp(-0.5 * ((uni - peak) / spread) ** 2)
                    ax.fill_between(uni, gauss,
                                    alpha=0.20, color=C['ann'])
                    ax.plot(uni, gauss,
                            color=C['ann'], linewidth=1.2, linestyle=':',
                            label="ANN weight" if ni == 0 else "_")

            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("μ", fontsize=9)
            ax.set_ylim(-0.05, 1.25)
            ax.grid(True, alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        # ── Legend panel (last subplot) ──────────────────────
        ax_leg = axes6_flat[5]
        ax_leg.axis('off')
        legend_items = []

        # Manual line
        legend_items.append(
            plt.Line2D([0], [0], color=C['manual'], linewidth=2,
                       label="Manual FIS (baseline)")
        )
        # Term dashes
        for term, col in [("Rendah/Sedikit/Rajin/Stabil", C['Rendah']),
                           ("Sedang/Cukup",                  C['Sedang']),
                           ("Tinggi/Banyak/Jarang/Rentan",   C['Tinggi'])]:
            legend_items.append(
                plt.Line2D([0], [0], color=col, linewidth=0.8,
                           linestyle='--', alpha=0.8, label=term)
            )
        if ga_ready:
            legend_items.append(
                plt.Line2D([0], [0], color=C['ga'], linewidth=2,
                           linestyle='--', label="GA-Tuned FIS")
            )
        if ann_ready:
            legend_items.append(
                plt.matplotlib.patches.Patch(
                    facecolor=C['ann'], alpha=0.35,
                    label="ANN Sensitivity Overlay"
                )
            )
        ax_leg.legend(handles=legend_items, loc='center', fontsize=9,
                      title="Keterangan", title_fontsize=10,
                      frameon=True, framealpha=0.9)

        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        st.divider()

        # ── Numerical delta table ────────────────────────────
        if ga_ready:
            st.markdown("#### Pergeseran Titik MF — Manual vs GA-Tuned")
            st.caption(
                "Nilai positif = GA menggeser titik ke kanan (lebih toleran); "
                "nilai negatif = menggeser ke kiri (lebih ketat)."
            )
            mf_params = get_ga_mf_params(st.session_state['ga_best_sol'])
            BASELINE_MANUAL = {
                'IPK':       {'Rendah_upper':2.2, 'Sedang_peak':2.5,  'Tinggi_lower':2.8,  'Rendah_inner':1.5},
                'Kehadiran': {'Jarang_upper':60,  'Cukup_peak':70,    'Rajin_lower':85,    'Jarang_inner':40},
                'MK_Gagal':  {'Sedikit_upper':3,  'Sedang_peak':4,    'Banyak_lower':7,    'Sedikit_inner':1, 'Banyak_upper':10},
                'StatusEkon':{'Rentan_mean':0.0,  'Stabil_mean':1.0},
                'Risiko':    {'Rendah_upper':45,  'Sedang_peak':50,   'Tinggi_lower':75,   'Rendah_lower':35, 'Tinggi_upper':75},
            }
            import pandas as pd
            delta_rows = []
            for var_name, params in mf_params.items():
                baseline_var = BASELINE_MANUAL.get(var_name, {})
                for param_name, ga_val in params.items():
                    base_val = baseline_var.get(param_name, float('nan'))
                    delta_rows.append({
                        "Variabel": var_name,
                        "Parameter MF": param_name,
                        "Manual (Baseline)": f"{base_val:.3f}",
                        "GA-Tuned": f"{ga_val:.4f}",
                        "Δ (GA − Manual)": f"{ga_val - base_val:+.4f}",
                    })
            st.dataframe(
                pd.DataFrame(delta_rows).set_index("Variabel"),
                use_container_width=True,
            )
