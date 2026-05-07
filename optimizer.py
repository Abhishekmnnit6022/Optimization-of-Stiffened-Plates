"""
Contains the genetic algorithm logic.
Sets up the optimization problem and constraints, evaluates candidates using the PINN,
and extracts the best designs.
"""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import tensorflow as tf
from tensorflow import keras

# ── Load original PINN model at startup (once) ────────────────────────────────
model  = keras.models.load_model('pinn_plate_model.keras')
x_mean = np.load('x_mean.npy')
x_std  = np.load('x_std.npy')

# Feature order (11 features) — matches x_mean.npy shape (11,):
# [plate_length, plate_breadth, tp, q_load, BC, nx, ny, dsx, wsx, dsy, wsy]


class StiffenedPlateProblem(Problem):
    """
    Defines the genetic algorithm problem, including limits for stiffener dimensions,
    objective functions, and constraints to ensure valid plate volumes.
    """
    def __init__(self, plate_length, plate_breadth, q_load, BC, tp_initial, **kwargs):
        self.plate_length  = float(plate_length)
        self.plate_breadth = float(plate_breadth)
        self.q_load        = float(q_load)
        self.BC            = int(BC)
        self.tp_initial    = float(tp_initial)

        super().__init__(
            n_var=6,
            n_obj=3,
            n_ieq_constr=12,
            xl=np.array([0,   0,  20,  5,  20,  5]),
            xu=np.array([12, 12, 180, 20, 180, 20]),
            **kwargs,
        )

        self.total_volume = self.plate_length * self.plate_breadth * self.tp_initial

    def _compute_tp(self, nx, ny, dsx, wsx, dsy, wsy):
        """
        Calculates the remaining plate thickness.
        Because total volume is strictly constrained, adding stiffeners means 
        we must subtract their volume from the base plate, making it thinner.
        """
        v_sx = nx * (self.plate_breadth * dsx * wsx)
        v_sy = ny * (self.plate_length  * dsy * wsy)
        v_stiff      = v_sx + v_sy
        v_plate_left = self.total_volume - v_stiff
        tp_new = v_plate_left / max(self.plate_length * self.plate_breadth, 1e-9)
        return tp_new, v_stiff, v_plate_left

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The core evaluation loop used by the Genetic Algorithm (NSGA-II).
        For each batch of candidate designs, it calculates their physical constraints,
        passes them through the PINN model for predictions, and returns the fitness scores.
        """
        n_samples = x.shape[0]
        nx  = np.round(x[:, 0]).astype(int)
        ny  = np.round(x[:, 1]).astype(int)
        dsx = x[:, 2]
        wsx = x[:, 3]
        dsy = x[:, 4]
        wsy = x[:, 5]

        tp, v_stiff, v_plate_left = self._compute_tp(nx, ny, dsx, wsx, dsy, wsy)

        spacing_x = np.where(nx > 0, self.plate_length  / (nx + 1), self.plate_length)
        spacing_y = np.where(ny > 0, self.plate_breadth / (ny + 1), self.plate_breadth)

        # ── 11-feature input — order MUST match training ───────────────────────
        # [plate_length, plate_breadth, tp, q_load, BC, nx, ny, dsx, wsx, dsy, wsy]
        X_input = np.column_stack([
            np.full(n_samples, self.plate_length),    # 0  plate_length
            np.full(n_samples, self.plate_breadth),   # 1  plate_breadth
            tp,                                       # 2  tp
            np.full(n_samples, self.q_load),          # 3  q_load
            np.full(n_samples, float(self.BC)),       # 4  BC
            nx.astype(float),                         # 5  nx
            ny.astype(float),                         # 6  ny
            dsx,                                      # 7  dsx
            wsx,                                      # 8  wsx
            dsy,                                      # 9  dsy
            wsy,                                      # 10 wsy
        ]).astype(np.float32)

        X_norm = (X_input - x_mean) / (x_std + 1e-8)
        pred   = model.predict(X_norm, verbose=0)    # shape (n_samples, 3)

        deflection = pred[:, 0]
        stress     = pred[:, 1]
        nat_freq   = pred[:, 2]

        F = np.zeros((n_samples, 3))
        F[:, 0] =  deflection
        F[:, 1] =  stress
        F[:, 2] = -nat_freq    # minimise negative = maximise frequency

        G = np.zeros((n_samples, 12))
        G[:, 0] = 4.0 - tp
        G[:, 1] = tp - self.tp_initial
        G[:, 2] = 75 - spacing_x
        G[:, 3] = 75 - spacing_y
        G[:, 4] = wsx - dsx
        G[:, 5] = wsy - dsy
        G[:, 6] = -v_plate_left
        G[:, 7] = v_stiff - 0.8 * self.total_volume
        G[:, 8] = self.total_volume * 0.05 - v_stiff
        
        # ── AI Sanity Constraints (Anti-Hallucination) ─────────────────────────
        G[:, 9]  = 0.05 - deflection  # Deflection must be >= 0.05 mm
        G[:, 10] = 10.0 - stress      # Stress must be >= 10.0 MPa
        G[:, 11] = 5.0 - nat_freq     # Frequency must be >= 5.0 Hz

        out["F"] = F
        out["G"] = G


def build_physics_heatmaps(length, breadth, q_load, bc, nx, ny, grid_n=42):
    """
    Generates a rough 2D surface matrix for deflection and stress so we can plot it 
    as a 3D terrain on the frontend.
    """
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)

    def_map = np.zeros((grid_n, grid_n), dtype=float)
    str_map = np.zeros((grid_n, grid_n), dtype=float)

    stiffness_x = np.ones(grid_n)
    stiffness_y = np.ones(grid_n)

    if nx > 0:
        for k in range(nx):
            loc = (k + 1) / (nx + 1)
            stiffness_x += 2.5 * np.exp(-((xs - loc) ** 2) / 0.002)
    if ny > 0:
        for k in range(ny):
            loc = (k + 1) / (ny + 1)
            stiffness_y += 2.5 * np.exp(-((ys - loc) ** 2) / 0.002)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            S = stiffness_x[j] * stiffness_y[i]

            if bc == 0:
                w = np.sin(np.pi * x) * np.sin(np.pi * y)
                s = np.sin(np.pi * x) * np.sin(np.pi * y)
            else:
                w = (1 - np.cos(2 * np.pi * x)) * (1 - np.cos(2 * np.pi * y)) / 4.0
                mx = np.cos(2 * np.pi * x) * (1 - np.cos(2 * np.pi * y))
                my = np.cos(2 * np.pi * y) * (1 - np.cos(2 * np.pi * x))
                s  = np.abs(mx + my) / 2.0 + 0.1 * w

            def_map[i, j] = w / S
            str_map[i, j] = s / np.sqrt(S)

    def_map = def_map / (np.max(def_map) + 1e-9)
    str_map = str_map / (np.max(str_map) + 1e-9)

    return def_map, str_map


def _predict_single(plate_length, plate_breadth, tp, q_load, BC,
                    nx, ny, dsx, wsx, dsy, wsy):
    """Call PINN for a single design point (used for candidate re-evaluation)."""
    X_input = np.array([[
        plate_length, plate_breadth, tp, q_load, float(BC),
        float(nx), float(ny), dsx, wsx, dsy, wsy
    ]], dtype=np.float32)
    X_norm = (X_input - x_mean) / (x_std + 1e-8)
    pred   = model.predict(X_norm, verbose=0)[0]
    return {
        'deflection': float(max(pred[0], 0.001)),
        'stress':     float(max(pred[1], 0.1)),
        'nat_freq':   float(max(pred[2], 1.0)),
    }


def run_optimization(bc, plate_length, plate_breadth, initial_thickness, q_load):
    """
    Sets up and runs the NSGA2 algorithm. It sorts through the generations, 
    picks out the top candidate, builds the heatmaps for it, and returns all the 
    data needed by the UI.
    """
    problem = StiffenedPlateProblem(
        plate_length=plate_length,
        plate_breadth=plate_breadth,
        q_load=q_load,
        BC=bc,
        tp_initial=initial_thickness,
    )

    algorithm = NSGA2(
        pop_size=90,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", 80), seed=1, verbose=False)

    if res.X is None:
        raise RuntimeError(
            "No feasible solution found for the given dimensions, thickness, and loading."
        )

    xs = np.atleast_2d(res.X)
    fs = np.atleast_2d(res.F)

    score = fs[:, 1] + 50.0 * fs[:, 0] - 0.02 * (-fs[:, 2])
    order = np.argsort(score)

    top_idx = [order[0]]
    for idx in np.argsort(fs[:, 0]):
        if idx not in top_idx:
            top_idx.append(idx)
            break
    for idx in np.argsort(fs[:, 1]):
        if idx not in top_idx:
            top_idx.append(idx)
            break
    for idx in np.argsort(fs[:, 2]):
        if idx not in top_idx:
            top_idx.append(idx)
            break
    for idx in order:
        if len(top_idx) >= 4:
            break
        if idx not in top_idx:
            top_idx.append(idx)

    def classify_pattern(nx, ny):
        if nx == 0 and ny == 0:
            return "flat"
        if ny == 0 or nx >= max(3, 2.5 * max(ny, 1)):
            return "transverse"
        if nx == 0 or ny >= max(3, 2.5 * max(nx, 1)):
            return "longitudinal"
        if plate_length > 1.8 * plate_breadth and nx >= 1.6 * max(ny, 1):
            return "transverse"
        if plate_breadth > 1.8 * plate_length and ny >= 1.6 * max(nx, 1):
            return "longitudinal"
        return "grid"

    candidates = []
    for rank, idx in enumerate(top_idx):
        x    = xs[idx]
        nx_  = int(np.round(x[0]))
        ny_  = int(np.round(x[1]))
        dsx_ = float(x[2])
        wsx_ = float(x[3])
        dsy_ = float(x[4])
        wsy_ = float(x[5])
        
        # ── Static Angles for Orthotropic Plates ──
        ang_x = 0.0
        ang_y = 90.0

        tp_, v_stiff_, v_plate_left_ = problem._compute_tp(nx_, ny_, dsx_, wsx_, dsy_, wsy_)
        spacing_x_ = plate_length  / (nx_ + 1) if nx_ > 0 else 0.0
        spacing_y_ = plate_breadth / (ny_ + 1) if ny_ > 0 else 0.0

        pred = _predict_single(
            plate_length, plate_breadth, float(tp_), q_load, bc,
            nx_, ny_, dsx_, wsx_, dsy_, wsy_
        )

        candidates.append({
            "rank":                   int(rank + 1),
            "pattern_type":           classify_pattern(nx_, ny_),
            "num_x":                  nx_,
            "num_y":                  ny_,
            "depth_x":                dsx_,
            "width_x":                wsx_,
            "depth_y":                dsy_,
            "width_y":                wsy_,
            "angle_x_deg":            ang_x,
            "angle_y_deg":            ang_y,
            "thickness":              float(tp_),
            "spacing_x":              spacing_x_,
            "spacing_y":              spacing_y_,
            "stiffener_volume":       float(v_stiff_),
            "plate_volume_remaining": float(v_plate_left_),
            "deflection":             pred["deflection"],
            "stress":                 pred["stress"],
            "frequency":              pred["nat_freq"],
            "score":                  float(score[idx]),
        })

    best    = candidates[0]
    def_map, str_map = build_physics_heatmaps(
        plate_length, plate_breadth, q_load, bc, best["num_x"], best["num_y"]
    )
    def_map = def_map * best["deflection"]
    str_map = str_map * best["stress"]

    pareto_points = [{"deflection": float(f[0]), "stress": float(f[1])} for f in fs]

    return {
        "pattern_type":           best["pattern_type"],
        "num_x":                  best["num_x"],
        "num_y":                  best["num_y"],
        "stiffener_length_x":     float(plate_breadth),
        "stiffener_length_y":     float(plate_length),
        "depth_x":                best["depth_x"],
        "width_x":                best["width_x"],
        "depth_y":                best["depth_y"],
        "width_y":                best["width_y"],
        "angle_x_deg":            best["angle_x_deg"],
        "angle_y_deg":            best["angle_y_deg"],
        "initial_thickness":      float(initial_thickness),
        "thickness":              best["thickness"],
        "length":                 float(plate_length),
        "breadth":                float(plate_breadth),
        "spacing_x":              best["spacing_x"],
        "spacing_y":              best["spacing_y"],
        "uniform_load":           float(q_load),
        "total_volume":           float(problem.total_volume),
        "stiffener_volume":       best["stiffener_volume"],
        "plate_volume_remaining": best["plate_volume_remaining"],
        "optimal_deflection":     best["deflection"],
        "optimal_stress":         best["stress"],
        "optimal_frequency":      best["frequency"],
        "candidates":             candidates,
        "pareto_points":          pareto_points,
        "deflection_heatmap":     def_map.tolist(),
        "stress_heatmap":         str_map.tolist(),
        "def_max":                float(best["deflection"]),
        "str_max":                float(best["stress"]),
    }