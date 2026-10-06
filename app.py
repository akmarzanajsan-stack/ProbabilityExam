import streamlit as st
import pandas as pd
from math import comb

st.set_page_config(page_title="Exam Predictor Pro", page_icon="🎓", layout="wide")


# ---------- Binomial distribution functions ----------
def pmf(n, k, p):
    """P(X = k) = C(n,k) * p^k * (1-p)^(n-k)"""
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def p_at_least(n, k, p):
    """P(X >= k)"""
    return sum(pmf(n, i, p) for i in range(k, n + 1))


def p_at_most(n, k, p):
    """P(X <= k)"""
    return sum(pmf(n, i, p) for i in range(0, k + 1))


def required_p(n, k, target, kind="at_least"):
    """Find p such that P(X >= k) = target (or P(X <= k) = target) by bisection.
    P(X >= k) grows with p, P(X <= k) falls with p."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if kind == "at_least":
            if p_at_least(n, k, mid) < target:
                lo = mid
            else:
                hi = mid
        else:
            if p_at_most(n, k, mid) > target:
                lo = mid
            else:
                hi = mid
    return (lo + hi) / 2


# ---------- Sidebar: shared exam parameters ----------
st.sidebar.header("⚙️ Exam Parameters")
n = st.sidebar.number_input("📝 Number of questions (n):", 1, 200, 10,
                            help="Total number of questions on the exam.")
k = st.sidebar.number_input("✅ Minimum correct to pass (k):", 0, 200, 8,
                            help="You pass if you answer at least k questions correctly.")
n, k = int(n), int(k)
if k > n:
    st.sidebar.error("k cannot be larger than n.")
    st.error("The minimum number of correct answers (k) cannot be larger than the number of questions (n).")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 What do these mean?")
st.sidebar.info(
    f"**n = {n}** → there are {n} questions.\n\n"
    f"**k = {k}** → you need at least {k} correct answers.\n\n"
    "**p** → the probability of answering ONE question correctly."
)

st.title("🎓 Exam Predictor Pro")
st.subheader("Binomial distribution: two questions, one app")
st.write("**Tab 1:** you know your skill *p*, find the chance to pass.  "
         "**Tab 2:** you know the chance you want, find the skill *p* you need.")

tab1, tab2 = st.tabs(["1️⃣ Find the probability", "2️⃣ Find the required p"])

# ---------- TAB 1: probability from p ----------
with tab1:
    p = st.slider("🎯 Probability of a correct answer (p):", 0.0, 1.0, 0.45, 0.01)
    kind = st.radio("📊 Calculation type:", ["P(X = k)", "P(X ≥ k)", "P(X ≤ k)"], index=1, horizontal=True)
    if kind == "P(X = k)":
        res, txt = pmf(n, k, p), f"exactly {k} correct answers"
    elif kind == "P(X ≥ k)":
        res, txt = p_at_least(n, k, p), f"AT LEAST {k} correct answers (PASS)"
    else:
        res, txt = p_at_most(n, k, p), f"AT MOST {k} correct answers"
    st.markdown(f"## 📊 {kind} = {res:.2%}")
    st.caption(f"Probability of getting {txt}")

    c1, c2, c3 = st.columns(3)
    c1.metric("E[X] — Expected correct", f"{n * p:.2f}", help="E[X] = n × p")
    c2.metric("Standard deviation", f"{(n * p * (1 - p)) ** 0.5:.2f}", help="σ = √(n·p·(1−p))")
    c3.metric("Your chance to pass", f"{p_at_least(n, k, p):.1%}")
    st.caption(f"Formula: n × p = {n} × {p:.2f} = {n * p:.2f}")

    dist = pd.DataFrame({"Correct answers (X)": range(n + 1),
                         "Probability": [pmf(n, i, p) for i in range(n + 1)]}).set_index("Correct answers (X)")
    st.bar_chart(dist)
    st.info(f"If 100 students with the same skill took this exam, about "
            f"**{round(100 * p_at_least(n, k, p))}** would pass.")

# ---------- TAB 2: p from target probability ----------
with tab2:
    kind2 = st.radio("Goal:", ["At least k correct (PASS)", "At most k correct"], horizontal=True)
    kk = "at_least" if kind2.startswith("At least") else "at_most"
    target = st.slider("🎯 Desired probability:", 0.01, 0.99, 0.50, 0.01, format="%.2f")

    if kk == "at_least" and k == 0:
        st.success("With k = 0 you always pass (probability 100%). Choose k ≥ 1.")
    else:
        rp = required_p(n, k, target, kk)
        st.markdown(f"## 🎯 Required p = {rp:.2%}")
        st.caption(f"To have a {target:.0%} chance of getting "
                   f"{'at least' if kk == 'at_least' else 'at most'} {k} correct answers out of {n}, "
                   f"you must answer each question correctly with probability about {rp:.1%}.")
        d1, d2 = st.columns(2)
        d1.metric("Required p", f"{rp:.1%}")
        d2.metric("Expected correct at this p", f"{n * rp:.2f} of {n}")

        rows = []
        for t in [0.5, 0.7, 0.9, 0.95, 0.99]:
            r = required_p(n, k, t, kk)
            rows.append({"Desired chance": f"{t:.0%}", "Required p": f"{r:.1%}", "Expected correct": f"{n * r:.2f}"})
        st.markdown("**Required p for different goals**")
        st.table(pd.DataFrame(rows).set_index("Desired chance"))

        f = p_at_least if kk == "at_least" else p_at_most
        curve = pd.DataFrame({"p": [i / 100 for i in range(101)],
                              "Probability": [f(n, k, i / 100) for i in range(101)]}).set_index("p")
        st.markdown("**Chance of success for every possible p**")
        st.line_chart(curve)
        st.info("Reading the curve: find your target on the vertical axis, move right until the curve, "
                "and look down to see the p you need.")

st.markdown("---")
st.markdown("### 📘 Formulas")
st.latex(r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
st.latex(r"P(X\ge k)=\sum_{i=k}^{n}\binom{n}{i}p^i(1-p)^{n-i}")
st.latex(r"E[X]=np,\qquad \sigma=\sqrt{np(1-p)}")
st.caption("The model assumes independent questions with the same probability p. Real exams are only approximately like this.")
