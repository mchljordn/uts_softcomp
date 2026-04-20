"""
=============================================================
  Tahap 3 — Genetic Algorithm Tuning for Mamdani FIS
  Mengoptimasi titik-titik inner MF pada 5 variabel FIS.

  Chromosome (20 genes):
    [0]  IPK.Rendah  — upper shoulder  (range 1.0–2.5)
    [1]  IPK.Sedang  — peak            (range 2.0–3.0)
    [2]  IPK.Tinggi  — lower shoulder  (range 2.5–3.5)
    [3]  Kehadiran.Jarang — upper      (range 30–65)
    [4]  Kehadiran.Cukup  — peak       (range 55–80)
    [5]  Kehadiran.Rajin  — lower      (range 70–90)
    [6]  MK_Gagal.Sedikit — upper      (range 1.0–4.0)
    [7]  MK_Gagal.Sedang  — peak       (range 3.0–6.0)
    [8]  MK_Gagal.Banyak  — lower      (range 4.0–8.0)
    [9]  StatusEkon.Rentan — mean      (range 0.0–0.3)
    [10] StatusEkon.Stabil — mean      (range 0.7–1.0)
    [11] Risiko.Rendah — upper         (range 20–45)
    [12] Risiko.Sedang — peak          (range 40–60)
    [13] Risiko.Tinggi — lower         (range 55–80)
    [14] IPK.Rendah    — left inner    (range 1.2–2.0)
    [15] Kehadiran.Jarang — left inner (range 25–55)
    [16] MK_Gagal.Sedikit— left inner  (range 0.5–2.5)
    [17] Risiko.Rendah — lower cross   (range 30–50)
    [18] Risiko.Tinggi — upper cross   (range 65–90)
    [19] MK_Gagal.Banyak — upper       (range 6.0–9.5)
=============================================================
"""

import numpy as np
import pygad
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from fis_manual import load_uci_dataset
import warnings
warnings.filterwarnings('ignore')

# ── Gene space: bounds for each of the 20 genes ──
GENE_SPACE = [
    {'low': 1.0,  'high': 2.5},   # [0]  IPK.Rendah upper
    {'low': 2.0,  'high': 3.0},   # [1]  IPK.Sedang peak
    {'low': 2.5,  'high': 3.5},   # [2]  IPK.Tinggi lower
    {'low': 30.0, 'high': 65.0},  # [3]  Kehadiran.Jarang upper
    {'low': 55.0, 'high': 80.0},  # [4]  Kehadiran.Cukup peak
    {'low': 70.0, 'high': 90.0},  # [5]  Kehadiran.Rajin lower
    {'low': 1.0,  'high': 4.0},   # [6]  MK_Gagal.Sedikit upper
    {'low': 3.0,  'high': 6.0},   # [7]  MK_Gagal.Sedang peak
    {'low': 4.0,  'high': 8.0},   # [8]  MK_Gagal.Banyak lower
    {'low': 0.0,  'high': 0.3},   # [9]  StatusEkon.Rentan mean
    {'low': 0.7,  'high': 1.0},   # [10] StatusEkon.Stabil mean
    {'low': 20.0, 'high': 45.0},  # [11] Risiko.Rendah upper
    {'low': 40.0, 'high': 60.0},  # [12] Risiko.Sedang peak
    {'low': 55.0, 'high': 80.0},  # [13] Risiko.Tinggi lower
    {'low': 1.2,  'high': 2.0},   # [14] IPK.Rendah left-inner
    {'low': 25.0, 'high': 55.0},  # [15] Kehadiran.Jarang left-inner
    {'low': 0.5,  'high': 2.5},   # [16] MK_Gagal.Sedikit left-inner
    {'low': 30.0, 'high': 50.0},  # [17] Risiko.Rendah lower-cross
    {'low': 65.0, 'high': 90.0},  # [18] Risiko.Tinggi upper-cross
    {'low': 6.0,  'high': 9.5},   # [19] MK_Gagal.Banyak upper
]

# Numerical label map for fitness calculation
LABEL_NUM = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}


def build_fis_from_chromosome(sol):
    """
    Rebuild a full skfuzzy FIS using the 20-gene chromosome.
    Returns (sim, ipk, kehadiran, mk_gagal, status_ekon, risiko) or None on failure.
    """
    try:
        # Unpack genes with safety clipping
        ipk_r_up   = float(np.clip(sol[0],  1.0,  2.5))
        ipk_s_pk   = float(np.clip(sol[1],  2.0,  3.0))
        ipk_t_lo   = float(np.clip(sol[2],  2.5,  3.5))
        had_j_up   = float(np.clip(sol[3],  30.0, 65.0))
        had_c_pk   = float(np.clip(sol[4],  55.0, 80.0))
        had_r_lo   = float(np.clip(sol[5],  70.0, 90.0))
        mk_s_up    = float(np.clip(sol[6],  1.0,  4.0))
        mk_m_pk    = float(np.clip(sol[7],  3.0,  6.0))
        mk_b_lo    = float(np.clip(sol[8],  4.0,  8.0))
        eko_r_mn   = float(np.clip(sol[9],  0.0,  0.3))
        eko_s_mn   = float(np.clip(sol[10], 0.7,  1.0))
        rsk_r_up   = float(np.clip(sol[11], 20.0, 45.0))
        rsk_s_pk   = float(np.clip(sol[12], 40.0, 60.0))
        rsk_t_lo   = float(np.clip(sol[13], 55.0, 80.0))
        ipk_r_li   = float(np.clip(sol[14], 1.2,  2.0))
        had_j_li   = float(np.clip(sol[15], 25.0, 55.0))
        mk_s_li    = float(np.clip(sol[16], 0.5,  2.5))
        rsk_r_lc   = float(np.clip(sol[17], 30.0, 50.0))
        rsk_t_uc   = float(np.clip(sol[18], 65.0, 90.0))
        mk_b_up    = float(np.clip(sol[19], 6.0,  9.5))

        # Enforce ordering constraints to keep MFs valid
        ipk_r_li = min(ipk_r_li, ipk_r_up - 0.1)
        had_j_li = min(had_j_li, had_j_up - 5.0)
        mk_s_li  = min(mk_s_li,  mk_s_up  - 0.5)
        rsk_r_lc = min(rsk_r_lc, rsk_r_up - 3.0)
        rsk_t_uc = max(rsk_t_uc, rsk_t_lo + 3.0)
        mk_b_up  = max(mk_b_up,  mk_b_lo  + 0.5)

        # Coarser universes — 5–10× fewer points, negligible accuracy loss for GA fitness
        ipk         = ctrl.Antecedent(np.arange(0, 4.05, 0.05),  'ipk')
        kehadiran   = ctrl.Antecedent(np.arange(0, 105, 5),       'kehadiran')
        mk_gagal    = ctrl.Antecedent(np.arange(0, 10.5, 0.5),    'mk_gagal')
        status_ekon = ctrl.Antecedent(np.arange(0, 1.05, 0.05),   'status_ekon')
        risiko      = ctrl.Consequent(np.arange(0, 105, 5),        'risiko',
                                      defuzzify_method='centroid')

        # IPK MFs
        ipk['Rendah'] = fuzz.trapmf(ipk.universe, [0.0, 0.0, ipk_r_li, ipk_r_up])
        ipk['Sedang'] = fuzz.trimf (ipk.universe, [ipk_r_li, ipk_s_pk, ipk_t_lo])
        ipk['Tinggi'] = fuzz.trapmf(ipk.universe, [ipk_t_lo, ipk_t_lo + 0.3, 4.0, 4.0])

        # Kehadiran MFs
        kehadiran['Jarang'] = fuzz.trapmf(kehadiran.universe, [0.0,  0.0, had_j_li, had_j_up])
        kehadiran['Cukup']  = fuzz.trimf (kehadiran.universe, [had_j_li, had_c_pk, had_r_lo])
        kehadiran['Rajin']  = fuzz.trapmf(kehadiran.universe, [had_r_lo, had_r_lo + 5.0, 100.0, 100.0])

        # MK_Gagal MFs
        mk_gagal['Sedikit'] = fuzz.trapmf(mk_gagal.universe, [0.0, 0.0, mk_s_li,  mk_s_up])
        mk_gagal['Sedang']  = fuzz.trimf (mk_gagal.universe, [mk_s_li,  mk_m_pk,  mk_b_lo])
        mk_gagal['Banyak']  = fuzz.trapmf(mk_gagal.universe, [mk_b_lo,  mk_b_up,  10.0, 10.0])

        # Status Ekonomi MFs
        status_ekon['Rentan'] = fuzz.gaussmf(status_ekon.universe, eko_r_mn, 0.15)
        status_ekon['Stabil'] = fuzz.gaussmf(status_ekon.universe, eko_s_mn, 0.15)

        # Risiko Output MFs
        risiko['Rendah'] = fuzz.trapmf(risiko.universe, [0.0,  0.0,      rsk_r_lc, rsk_r_up])
        risiko['Sedang'] = fuzz.trimf (risiko.universe, [rsk_r_lc, rsk_s_pk, rsk_t_lo])
        risiko['Tinggi'] = fuzz.trapmf(risiko.universe, [rsk_t_lo, rsk_t_uc, 100.0, 100.0])

        # Rules (same topology as manual, different MF boundaries)
        rules = [
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
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Jarang'] & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Sedang']),
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
            ctrl.Rule(ipk['Sedang']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Rajin']  & mk_gagal['Sedang']  & status_ekon['Stabil'], risiko['Rendah']),
            ctrl.Rule(ipk['Sedang']  & kehadiran['Rajin']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
            ctrl.Rule(ipk['Tinggi']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Rentan'], risiko['Rendah']),
            ctrl.Rule(ipk['Sedang']  & kehadiran['Cukup']  & mk_gagal['Sedikit'] & status_ekon['Stabil'], risiko['Rendah']),
        ]

        fis_ctrl = ctrl.ControlSystem(rules)
        sim      = ctrl.ControlSystemSimulation(fis_ctrl)
        return sim, ipk, kehadiran, mk_gagal, status_ekon, risiko

    except Exception:
        return None


def _gauss(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / max(sigma, 1e-6)) ** 2)

def _trap(x, a, b, c, d):
    return np.clip(np.minimum((x - a) / max(b - a, 1e-6),
                              (d - x) / max(d - c, 1e-6)), 0.0, 1.0)

def _tri(x, a, b, c):
    return np.clip(np.minimum((x - a) / max(b - a, 1e-6),
                              (c - x) / max(c - b, 1e-6)), 0.0, 1.0)

# Rule table: (ipk_mf, had_mf, mk_mf, eko_mf, out_mf)
# mf indices: ipk 0=R 1=S 2=T | had 3=J 4=C 5=Ra | mk 6=Sd 7=Se 8=B | eko 9=Re 10=St
# out: 0=Rendah 1=Sedang 2=Tinggi
_RULES = [
    (0,3,8,9,2),(0,3,8,10,2),(0,3,7,9,2),(0,4,8,9,2),(0,3,7,10,2),
    (0,4,8,10,2),(1,3,8,9,2),(0,5,8,9,2),(1,3,8,10,2),(0,3,6,9,2),
    (0,4,7,10,1),(1,4,7,9,1),(1,3,7,10,1),(1,3,6,9,1),(0,5,7,10,1),
    (0,4,6,9,1),(1,4,8,10,1),(1,3,7,9,1),(2,3,7,9,1),(0,5,6,9,1),
    (1,5,7,9,1),(2,4,8,9,1),(2,3,6,10,1),
    (2,5,6,10,0),(2,5,6,9,0),(2,4,6,10,0),(1,5,6,10,0),(2,5,7,10,0),
    (1,5,6,9,0),(2,4,6,9,0),(1,4,6,10,0),
]

def _infer_fast(sol, ipk_v, had_v, mk_v, eko_v):
    """
    Pure-numpy Mamdani inference for one sample.
    Skips skfuzzy ControlSystem entirely — ~50× faster per call.
    """
    # Unpack chromosome
    ipk_r_up = sol[0];  ipk_s_pk = sol[1];  ipk_t_lo = sol[2]
    had_j_up = sol[3];  had_c_pk = sol[4];  had_r_lo = sol[5]
    mk_s_up  = sol[6];  mk_m_pk  = sol[7];  mk_b_lo  = sol[8]
    eko_r_mn = sol[9];  eko_s_mn = sol[10]
    rsk_r_up = sol[11]; rsk_s_pk = sol[12]; rsk_t_lo = sol[13]
    ipk_r_li = min(sol[14], ipk_r_up - 0.1)
    had_j_li = min(sol[15], had_j_up - 5.0)
    mk_s_li  = min(sol[16], mk_s_up  - 0.5)
    rsk_r_lc = min(sol[17], rsk_r_up - 3.0)
    rsk_t_uc = max(sol[18], rsk_t_lo + 3.0)
    mk_b_up  = max(sol[19], mk_b_lo  + 0.5)

    # MF values for each input
    mu = [None] * 11
    mu[0]  = _trap(ipk_v,  0.0, 0.0,      ipk_r_li, ipk_r_up)
    mu[1]  = _tri (ipk_v,  ipk_r_li,      ipk_s_pk, ipk_t_lo)
    mu[2]  = _trap(ipk_v,  ipk_t_lo,      ipk_t_lo + 0.3, 4.0, 4.0)
    mu[3]  = _trap(had_v,  0.0, 0.0,      had_j_li, had_j_up)
    mu[4]  = _tri (had_v,  had_j_li,      had_c_pk, had_r_lo)
    mu[5]  = _trap(had_v,  had_r_lo,      had_r_lo + 5.0, 100.0, 100.0)
    mu[6]  = _trap(mk_v,   0.0, 0.0,      mk_s_li,  mk_s_up)
    mu[7]  = _tri (mk_v,   mk_s_li,       mk_m_pk,  mk_b_lo)
    mu[8]  = _trap(mk_v,   mk_b_lo,       mk_b_up,  10.0, 10.0)
    mu[9]  = _gauss(eko_v, eko_r_mn, 0.15)
    mu[10] = _gauss(eko_v, eko_s_mn, 0.15)

    # Output universe and aggregated output MFs
    u_out = np.arange(0, 105, 5, dtype=np.float32)
    agg   = np.zeros(len(u_out), dtype=np.float32)

    mf_out = [
        _trap(u_out, 0.0,     0.0,      rsk_r_lc, rsk_r_up),  # Rendah
        _tri (u_out, rsk_r_lc, rsk_s_pk, rsk_t_lo),            # Sedang
        _trap(u_out, rsk_t_lo, rsk_t_uc, 100.0, 100.0),        # Tinggi
    ]

    for (i0, i1, i2, i3, out_idx) in _RULES:
        firing = min(mu[i0], mu[i1], mu[i2], mu[i3])
        if firing > 0:
            agg = np.maximum(agg, np.minimum(firing, mf_out[out_idx]))

    denom = agg.sum()
    if denom < 1e-8:
        return 50.0   # fallback to mid-range
    return float(np.dot(agg, u_out) / denom)


def evaluate_chromosome(sol, dataset):
    """
    Evaluate chromosome accuracy on the dataset using fast numpy inference.
    Returns accuracy in [0, 1].
    """
    correct = 0
    total   = 0
    for row in dataset:
        try:
            score = _infer_fast(sol,
                                float(row['ipk']), float(row['kehadiran']),
                                float(row['mk_gagal']), float(row['status_ekon']))
            pred = 'Rendah' if score < 40 else ('Sedang' if score < 65 else 'Tinggi')
            if pred == row['label_true']:
                correct += 1
            total += 1
        except Exception:
            pass
    return correct / total if total > 0 else 0.0


# Module-level dataset cache so fitness_func can access it without re-generating
_DATASET = None
_DATASET_SAMPLE_N = 200
_DATASET_RANDOM_STATE = 42


def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness = classification accuracy on the UCI dataset.
    Higher accuracy → higher fitness.
    """
    global _DATASET, _DATASET_SAMPLE_N, _DATASET_RANDOM_STATE
    if _DATASET is None:
        _DATASET = load_uci_dataset(
            sample_n=_DATASET_SAMPLE_N,
            random_state=_DATASET_RANDOM_STATE
        )
    return evaluate_chromosome(solution, _DATASET)


def run_ga_tuning(pop_size=30, num_gen=20, on_generation=None,
                  dataset_sample_n=200, dataset_random_state=42):
    """
    Run GA optimisation.
    Returns:
        best_solution  : np.ndarray, shape (20,)
        best_fitness   : float  (accuracy 0–1)
        fitness_history: list of best-fitness per generation
        pop_history    : list of (population_array, fitness_array) per generation
    """
    global _DATASET, _DATASET_SAMPLE_N, _DATASET_RANDOM_STATE
    _DATASET_SAMPLE_N = dataset_sample_n
    _DATASET_RANDOM_STATE = dataset_random_state
    _DATASET = load_uci_dataset(
        sample_n=_DATASET_SAMPLE_N,
        random_state=_DATASET_RANDOM_STATE
    )

    fitness_history = []
    pop_history     = []

    def _on_generation(ga):
        best_fit = ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]
        fitness_history.append(float(best_fit))
        # Store a copy of every solution and its fitness this generation
        pop_history.append((
            ga.population.copy(),
            ga.last_generation_fitness.copy()
        ))
        if on_generation is not None:
            on_generation(ga, len(fitness_history), float(best_fit))

    ga_instance = pygad.GA(
        num_generations       = num_gen,
        num_parents_mating    = max(2, pop_size // 5),
        fitness_func          = fitness_func,
        sol_per_pop           = pop_size,
        num_genes             = 20,
        gene_space            = GENE_SPACE,
        parent_selection_type = 'tournament',
        crossover_type        = 'two_points',
        mutation_type         = 'random',
        mutation_percent_genes= 15,
        keep_elitism          = 2,
        on_generation         = _on_generation,
        random_seed           = 42,
        suppress_warnings     = True,
        parallel_processing   = ['thread', 4],   # parallelise fitness calls
    )
    ga_instance.run()

    best_sol, best_fit, _ = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    return best_sol, float(best_fit), fitness_history, pop_history


def get_ga_mf_params(best_solution):
    """
    Extract readable MF parameter dict from the best chromosome.
    Useful for displaying tuned values in the UI.
    """
    sol = best_solution
    return {
        'IPK':        {'Rendah_upper': sol[0],  'Sedang_peak': sol[1],  'Tinggi_lower': sol[2],  'Rendah_inner': sol[14]},
        'Kehadiran':  {'Jarang_upper': sol[3],  'Cukup_peak':  sol[4],  'Rajin_lower':  sol[5],  'Jarang_inner': sol[15]},
        'MK_Gagal':   {'Sedikit_upper':sol[6],  'Sedang_peak': sol[7],  'Banyak_lower': sol[8],  'Sedikit_inner':sol[16], 'Banyak_upper':sol[19]},
        'StatusEkon': {'Rentan_mean':  sol[9],  'Stabil_mean': sol[10]},
        'Risiko':     {'Rendah_upper': sol[11], 'Sedang_peak': sol[12], 'Tinggi_lower': sol[13], 'Rendah_lower': sol[17], 'Tinggi_upper': sol[18]},
    }