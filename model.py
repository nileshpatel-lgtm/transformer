"""Optimisation core for UGVCL preventive-maintenance scheduling (EM618).

Binary MILP:  x[i,t] = 1 if transformer i is maintained in month t.
  min  sum c_i x_it + sum K R_i (1 - eff * y_i)            (+ tiny tie-break term)
  s.t. sum_t x_it = y_i                       (each unit at most once)
       sum_it c_i x_it <= Budget
       sum_i  h_i x_it <= Hours   for every month t
       sum_i  x_it     <= MaxJobs for every month t
       sum_t  x_it     >= 1       for every i with criticality >= threshold
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

REQUIRED_COLS = ["Transformer", "Age", "Peak_Load_pct", "Failures_24m",
                 "Criticality", "PM_Cost", "PM_Hours"]


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def compute_risk(df, w_age=0.25, w_load=0.30, w_fail=0.25, w_crit=0.20):
    """Min-max normalise the four indicators to [0,1] and combine (Eq. 4)."""
    out = df.copy()
    tot = w_age + w_load + w_fail + w_crit
    out["Risk"] = (w_age * _minmax(out["Age"]) + w_load * _minmax(out["Peak_Load_pct"])
                   + w_fail * _minmax(out["Failures_24m"])
                   + w_crit * _minmax(out["Criticality"])) / tot
    return out


def solve(df, T=6, budget=700000.0, hours_cap=34.0, max_jobs=4, eff=0.70,
          K=600000.0, crit_threshold=4, tiebreak="risk_first", relax=False):
    """Solve the MILP (or its LP relaxation if relax=True).
    Returns dict with x matrix, objective (without constant), status."""
    N = len(df)
    c = df["PM_Cost"].to_numpy(float)
    h = df["PM_Hours"].to_numpy(float)
    R = df["Risk"].to_numpy(float)
    crit = df["Criticality"].to_numpy()
    nx = N * T
    nv = nx + N + 1            # x (N*T), y (N), m (workload variable)
    xi = lambda i, t: i * T + t
    yi = lambda i: nx + i
    mi = nx + N

    obj = np.zeros(nv)
    eps = 1.0                  # rupee-scale tie-break, far below any real trade-off
    for i in range(N):
        for t in range(T):
            obj[xi(i, t)] = c[i]
            if tiebreak == "risk_first":
                obj[xi(i, t)] += eps * (t + 1) * R[i]
        obj[yi(i)] = -K * R[i] * eff
    if tiebreak == "balance":
        obj[mi] = 50.0

    rows, lo, hi = [], [], []
    A = lil_matrix((N + 1 + 2 * T + T + N, nv))
    r = 0
    for i in range(N):                      # sum_t x_it - y_i = 0
        for t in range(T):
            A[r, xi(i, t)] = 1
        A[r, yi(i)] = -1
        lo.append(0); hi.append(0); r += 1
    for i in range(N):                      # budget
        for t in range(T):
            A[r, xi(i, t)] = c[i]
    lo.append(-np.inf); hi.append(budget); r += 1
    for t in range(T):                      # hours
        for i in range(N):
            A[r, xi(i, t)] = h[i]
        lo.append(-np.inf); hi.append(hours_cap); r += 1
    for t in range(T):                      # job count
        for i in range(N):
            A[r, xi(i, t)] = 1
        lo.append(-np.inf); hi.append(max_jobs); r += 1
    for t in range(T):                      # workload balance helper: sum_i x_it - m <= 0
        for i in range(N):
            A[r, xi(i, t)] = 1
        A[r, mi] = -1
        lo.append(-np.inf); hi.append(0); r += 1
    for i in range(N):                      # criticality coverage
        if crit[i] >= crit_threshold:
            for t in range(T):
                A[r, xi(i, t)] = 1
            lo.append(1); hi.append(np.inf); r += 1
    A = A[:r].tocsr()

    integrality = np.zeros(nv) if relax else np.concatenate([np.ones(nx + N), [0]])
    ub = np.ones(nv); ub[mi] = max(max_jobs, 1)
    res = milp(obj, constraints=LinearConstraint(A, lo, hi),
               integrality=integrality, bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": 30, "mip_rel_gap": 1e-9})
    if res.x is None:
        return {"ok": False, "message": res.message}
    x = res.x[:nx].reshape(N, T)
    y = res.x[nx:nx + N]
    # true objective (no tie-break): direct cost + residual consequence
    direct = float((c[:, None] * x).sum())
    resid = float((K * R * (1 - eff * y)).sum())
    return {"ok": True, "x": np.round(x) if not relax else x, "y": y,
            "direct": direct, "residual": resid, "total": direct + resid,
            "var_obj": direct - float((K * R * eff * y).sum()),
            "message": res.message}


def schedule_table(df, x):
    rows = []
    for i in range(len(df)):
        ts = np.where(x[i] > 0.5)[0]
        if len(ts):
            rows.append({"Month": int(ts[0]) + 1, "Transformer": df.loc[i, "Transformer"],
                         "Risk": round(df.loc[i, "Risk"], 3),
                         "Criticality": int(df.loc[i, "Criticality"]),
                         "PM_Cost": df.loc[i, "PM_Cost"], "PM_Hours": df.loc[i, "PM_Hours"]})
    out = pd.DataFrame(rows)
    return out.sort_values(["Month", "Risk"], ascending=[True, False]).reset_index(drop=True) if len(out) else out


def baseline(df, T=6, per_month=3, eff=0.70, K=600000.0):
    """Age-based calendar rule: oldest units first, `per_month` jobs per month."""
    order = df.sort_values(["Age", "Transformer"], ascending=[False, False]).index[: T * per_month]  # ties: higher ID first
    sel = df.loc[order]
    direct = float(sel["PM_Cost"].sum())
    y = df.index.isin(order).astype(float)
    resid = float((K * df["Risk"].to_numpy() * (1 - eff * y)).sum())
    return {"jobs": len(sel), "direct": direct, "residual": resid, "total": direct + resid,
            "selected": sel["Transformer"].tolist()}
