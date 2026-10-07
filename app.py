"""EM618 - UGVCL Preventive Maintenance Decision-Support Dashboard (Streamlit)."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from model import REQUIRED_COLS, compute_risk, solve, schedule_table, baseline

st.set_page_config(page_title="UGVCL PM Scheduler", page_icon="⚡", layout="wide")
inr = lambda v: f"₹{v:,.0f}"

st.title("⚡ Preventive Maintenance Scheduling – Distribution Transformers")
st.caption("EM618 Optimization · Final Decision Report · Nilesh Prahladbhai Patel (25280046) · IIT Gandhinagar")
st.warning("Demonstration uses a **synthetic pilot dataset** (30 transformers). "
           "It is not verified UGVCL asset data. The output is decision support, not a maintenance order.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Transformer CSV (optional)", type="csv")
    st.caption("Columns: " + ", ".join(REQUIRED_COLS))
    st.header("Resources")
    T = st.slider("Planning horizon (months)", 3, 12, 6)
    budget = st.number_input("Budget (₹)", 100000, 5000000, 700000, 10000)
    hours = st.number_input("Technician-hours per month", 5.0, 200.0, 34.0, 1.0)
    max_jobs = st.number_input("Max jobs per month", 1, 20, 4)
    crit_thr = st.slider("Mandatory criticality ≥", 1, 5, 4)
    st.header("Model assumptions")
    eff = st.slider("PM risk reduction", 0.0, 1.0, 0.70, 0.05)
    K = st.number_input("Failure-consequence scale K (₹ per risk unit)", 0, 5000000, 600000, 50000)
    st.subheader("Risk weights")
    w_age = st.slider("Age", 0.0, 1.0, 0.25, 0.05)
    w_load = st.slider("Peak loading", 0.0, 1.0, 0.30, 0.05)
    w_fail = st.slider("Failure history", 0.0, 1.0, 0.25, 0.05)
    w_crit = st.slider("Criticality", 0.0, 1.0, 0.20, 0.05)
    if abs(w_age + w_load + w_fail + w_crit - 1) > 1e-9:
        st.caption("Weights are re-scaled to sum to 1.")
    tb = st.radio("Month assignment", ["Highest risk first", "Balance workload"])
    tiebreak = "risk_first" if tb == "Highest risk first" else "balance"

# ------------------------------------------------------------------ data
if up is not None:
    raw = pd.read_csv(up)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
else:
    raw = pd.read_csv("transformers_pilot.csv")
if w_age + w_load + w_fail + w_crit == 0:
    st.error("At least one risk weight must be positive."); st.stop()

df = compute_risk(raw.reset_index(drop=True), w_age, w_load, w_fail, w_crit)

kw = dict(T=T, budget=float(budget), hours_cap=float(hours), max_jobs=int(max_jobs),
          eff=eff, K=float(K), crit_threshold=crit_thr)
sol = solve(df, tiebreak=tiebreak, **kw)
if not sol["ok"]:
    st.error("No feasible plan under these limits (e.g. budget/hours too small to cover all "
             f"criticality ≥ {crit_thr} units). Relax a constraint.")
    st.stop()
sched = schedule_table(df, sol["x"])
base = baseline(df, T=T, per_month=3, eff=eff, K=float(K))
net = base["total"] - sol["total"]

# ------------------------------------------------------------------ KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs selected", len(sched))
c2.metric("PM spend", inr(sol["direct"]), f"{inr(budget - sol['direct'])} budget left", delta_color="off")
c3.metric("Expected total cost", inr(sol["total"]))
c4.metric("Net value vs age-based baseline", inr(net))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📅 Schedule", "📊 Risk ranking", "🔧 Resource use & checks", "📈 Sensitivity", "🧮 Solution quality"])

# ------------------------------------------------------------------ schedule
with tab1:
    st.subheader("Optimised month-wise schedule")
    show = sched.copy()
    show["PM_Cost"] = show["PM_Cost"].map(inr)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download schedule (CSV)", sched.to_csv(index=False).encode(),
                       "pm_schedule.csv", "text/csv")
    grid = pd.DataFrame(0, index=df["Transformer"], columns=[f"M{t+1}" for t in range(T)])
    grid.values[:] = sol["x"].astype(int)
    st.markdown("**Assignment matrix** (1 = maintained in that month)")
    st.dataframe(grid.style.map(lambda v: "background-color:#1f77b4;color:white" if v else ""),
                 use_container_width=True, height=300)
    st.subheader("Comparison with age-based calendar baseline")
    cmp_ = pd.DataFrame({
        "Metric": ["Maintenance jobs", "PM expenditure", "Expected total cost"],
        "Baseline (3 oldest per month)": [base["jobs"], inr(base["direct"]), inr(base["total"])],
        "Optimised": [len(sched), inr(sol["direct"]), inr(sol["total"])]})
    st.dataframe(cmp_, hide_index=True, use_container_width=True)
    if base["direct"] > budget:
        st.caption("Note: the baseline rule ignores the budget and exceeds it.")

# ------------------------------------------------------------------ ranking
with tab2:
    st.subheader("Composite risk score (Eq. 4)")
    rk = df.sort_values("Risk", ascending=False).copy()
    rk["Selected"] = rk["Transformer"].isin(sched["Transformer"]).map({True: "Yes", False: "No"})
    st.bar_chart(rk.set_index("Transformer")["Risk"])
    st.dataframe(rk[["Transformer", "Age", "Peak_Load_pct", "Failures_24m", "Criticality",
                     "Risk", "Selected"]].round(3), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ resources
with tab3:
    mj = sched.groupby("Month").size().reindex(range(1, T + 1), fill_value=0)
    mh = sched.groupby("Month")["PM_Hours"].sum().reindex(range(1, T + 1), fill_value=0.0)
    left, right = st.columns(2)
    with left:
        st.markdown("**Jobs per month**")
        st.bar_chart(pd.DataFrame({"Jobs": mj.values}, index=[f"M{m}" for m in mj.index]))
    with right:
        st.markdown(f"**Technician-hours per month** (limit {hours:g})")
        st.bar_chart(pd.DataFrame({"Hours": mh.values}, index=[f"M{m}" for m in mh.index]))
    crit_ids = df.loc[df["Criticality"] >= crit_thr, "Transformer"]
    covered = int(crit_ids.isin(sched["Transformer"]).sum())
    checks = pd.DataFrame([
        ["Budget", inr(sol["direct"]), f"≤ {inr(budget)}", sol["direct"] <= budget + 1e-6],
        ["Max monthly jobs", int(mj.max()), f"≤ {int(max_jobs)}", mj.max() <= max_jobs],
        ["Max monthly hours", round(float(mh.max()), 1), f"≤ {hours:g}", mh.max() <= hours + 1e-6],
        [f"Criticality ≥ {crit_thr} coverage", f"{covered} / {len(crit_ids)}", "All covered", covered == len(crit_ids)],
        ["Each unit at most once", int(sol["x"].sum(axis=1).max()), "≤ 1", sol["x"].sum(axis=1).max() <= 1],
    ], columns=["Constraint", "Observed", "Requirement", "Pass"])
    checks["Status"] = checks["Pass"].map({True: "PASS", False: "FAIL"})
    st.dataframe(checks.drop(columns="Pass"), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ sensitivity
with tab4:
    st.subheader("Budget sensitivity")
    lo_b, hi_b = st.slider("Budget range (₹ lakh)", 3.0, 12.0, (5.5, 8.0), 0.5)
    rows = []
    for b in np.arange(lo_b, hi_b + 1e-9, 0.5):
        s = solve(df, tiebreak="none", **{**kw, "budget": b * 100000})
        if s["ok"]:
            rows.append({"Budget (₹ lakh)": round(b, 1), "Jobs": int(s["x"].sum()),
                         "PM spend": s["direct"], "Expected total cost": round(s["total"])})
    if rows:
        sens = pd.DataFrame(rows)
        st.line_chart(sens.set_index("Budget (₹ lakh)")[["Jobs"]])
        st.dataframe(sens.assign(**{"PM spend": sens["PM spend"].map(inr),
                                    "Expected total cost": sens["Expected total cost"].map(inr)}),
                     hide_index=True, use_container_width=True)
    else:
        st.info("No feasible budget in this range.")

# ------------------------------------------------------------------ solution quality
with tab5:
    st.subheader("MILP vs LP-relaxation lower bound")
    rel = solve(df, relax=True, tiebreak="none", **kw)
    mip = solve(df, tiebreak="none", **kw)
    if rel["ok"] and mip["ok"]:
        gap = (mip["var_obj"] - rel["var_obj"]) / abs(rel["var_obj"]) * 100
        a, b_, c_ = st.columns(3)
        a.metric("LP relaxation objective", f"{rel['var_obj']:,.1f}")
        b_.metric("Integer objective", f"{mip['var_obj']:,.1f}")
        c_.metric("Gap", f"{gap:.2f}%")
        st.caption("Objective shown without the constant term K·ΣR (as in Section 4.1 of the report). "
                   "The LP bound is a lower bound for this minimisation problem.")
    st.markdown("**Model:** binary MILP solved exactly with HiGHS (`scipy.optimize.milp`).")
    st.latex(r"\min Z=\sum_{i,t}c_i x_{it}+\sum_i K R_i(1-\eta y_i)")
    st.latex(r"\sum_t x_{it}=y_i,\ \sum c_i x_{it}\le B,\ \sum_i h_i x_{it}\le H,\ \sum_i x_{it}\le J,\ \sum_t x_{it}\ge1\ (C_i\ge4)")
