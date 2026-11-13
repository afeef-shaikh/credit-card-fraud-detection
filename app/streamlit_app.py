"""Fraud loss simulator: the cost-sensitive fraud model from notebook 04, made interactive.

All model outputs are precomputed by app/build_artifacts.py, so the app needs no ML
libraries at runtime and loads instantly.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
REPO_URL = "https://github.com/afeef-shaikh/credit-card-fraud-detection"

# Design tokens
INK, MUTED, LINE, PAPER, PANEL = "#14231F", "#5E6E69", "#DCE2DE", "#F6F7F4", "#FFFFFF"
TEAL, CRIMSON, AMBER, GRAY = "#00806A", "#C0263F", "#B57A12", "#8A9692"
FONT = "Schibsted Grotesk, Helvetica Neue, Arial, sans-serif"

st.set_page_config(page_title="Fraud loss simulator", page_icon="💳", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input {{ font-family: {FONT}; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}
.num, .ledger td, .facts b {{ font-variant-numeric: tabular-nums; }}

.byline {{ color: {MUTED}; font-size: 0.95rem; margin: 0 0 0.9rem 0; }}
.byline a {{ color: {MUTED}; text-decoration: underline; text-underline-offset: 3px; }}
h1.hero {{ font-size: clamp(1.9rem, 3.6vw, 3.05rem); line-height: 1.08; letter-spacing: -0.022em;
           font-weight: 700; color: {INK}; max-width: 25ch; margin: 0 0 1.1rem 0; padding: 0; }}
p.lede {{ font-size: 1.12rem; line-height: 1.55; color: {INK}; max-width: 62ch; margin: 0 0 1.8rem 0; }}

.facts {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0 2.2rem;
          border-top: 2px solid {INK}; padding-top: 0.9rem; margin-bottom: 2.4rem; }}
.facts div {{ font-size: 0.98rem; line-height: 1.45; color: {MUTED}; }}
.facts b {{ display: block; font-size: 1.65rem; line-height: 1.15; color: {INK}; font-weight: 600;
            letter-spacing: -0.01em; margin-bottom: 0.2rem; }}
@media (max-width: 640px) {{
  .facts {{ grid-template-columns: 1fr; gap: 1rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.55rem 0; font-size: 1.02rem; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.02rem; font-weight: 500; }}

.readout {{ font-size: 1.08rem; line-height: 1.6; color: {INK}; max-width: 64ch; margin: 0.2rem 0 1rem 0; }}
.readout b {{ font-weight: 600; }}
.note {{ font-size: 0.9rem; line-height: 1.5; color: {MUTED}; max-width: 70ch; }}
.counts {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.6rem 1.4rem; margin: 0.4rem 0 0.6rem 0; }}
.counts div {{ border-left: 3px solid var(--c); padding: 0.1rem 0 0.1rem 0.7rem; color: {MUTED}; font-size: 0.88rem; }}
.counts b {{ display: block; color: {INK}; font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
@media (max-width: 640px) {{ .counts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

.ledger-wrap {{ max-height: 520px; overflow: auto; border: 1px solid {LINE}; border-radius: 6px; background: {PANEL}; }}
table.ledger {{ border-collapse: collapse; width: 100%; font-size: 0.93rem; }}
.ledger th {{ position: sticky; top: 0; background: {PANEL}; text-align: right; font-weight: 500;
              color: {MUTED}; padding: 0.6rem 0.9rem; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
.ledger td {{ text-align: right; padding: 0.5rem 0.9rem; border-bottom: 1px solid #EEF1EE; white-space: nowrap; color: {INK}; }}
.ledger th:first-child, .ledger td:first-child, .ledger th:nth-child(2), .ledger td:nth-child(2) {{ text-align: left; }}
.ledger tr.is-fraud td:first-child {{ box-shadow: inset 3px 0 0 {CRIMSON}; }}
.ledger.plain th:nth-child(2), .ledger.plain td:nth-child(2) {{ text-align: right; }}
.ledger.plain th, .ledger.plain td {{ padding: 0.5rem 0.6rem; white-space: normal; }}
.tag {{ display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px; font-size: 0.82rem; font-weight: 500; }}
.tag.fraud {{ background: #F8E3E6; color: #8E1C2F; }}
.tag.legit {{ background: #E9EEEC; color: {MUTED}; }}

.txn {{ border-top: 2px solid {INK}; padding-top: 0.8rem; }}
.txn .amt {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.015em; color: {INK}; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.txn dl {{ display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 1rem; margin: 0.9rem 0 1rem 0; font-size: 0.95rem; }}
.txn dt {{ color: {MUTED}; }} .txn dd {{ margin: 0; color: {INK}; font-variant-numeric: tabular-nums; }}

h3.sec {{ font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; margin: 1.6rem 0 0.35rem 0; color: {INK}; }}
.footer {{ border-top: 1px solid {LINE}; margin-top: 3rem; padding-top: 1rem; color: {MUTED}; font-size: 0.92rem; }}
.footer a {{ color: {INK}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    test = pd.read_parquet(DATA / "test.parquet")
    oof = pd.read_parquet(DATA / "oof.parquet")
    shap_df = pd.read_parquet(DATA / "shap_subset.parquet")
    meta = json.loads((DATA / "meta.json").read_text())
    test["exp_loss"] = test["prob"] * test["Amount"]
    return test, oof, shap_df, meta


test, oof, shap_df, meta = load()
W = meta["real_legit_weight"]
y, amt, prob = test["Class"].to_numpy(), test["Amount"].to_numpy(), test["prob"].to_numpy()
NO_MODEL = float(amt[y == 1].sum())
CUTOFFS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def outcome(y_, a_, p_, cut, review):
    """Counts and € for one cutoff. False alarms are scaled to real-world volume."""
    f = p_ >= cut
    tp, fn = int(((y_ == 1) & f).sum()), int(((y_ == 1) & ~f).sum())
    fp = int(((y_ == 0) & f).sum())
    fp_real = fp * W
    missed_eur = float(a_[(y_ == 1) & ~f].sum())
    review_eur = (tp + fp_real) * review
    return dict(caught=tp, missed=fn, fp=fp, fp_real=fp_real, alerts_real=tp + fp_real,
                caught_eur=float(a_[(y_ == 1) & f].sum()), missed_eur=missed_eur,
                review_eur=review_eur, total=missed_eur + review_eur)


@st.cache_data
def best_cutoff_from_past(review):
    """Walk-forward choice: cheapest cutoff on out-of-fold predictions from earlier data."""
    yo, ao, po = oof["Class"].to_numpy(), oof["Amount"].to_numpy(), oof["prob"].to_numpy()
    costs = [outcome(yo, ao, po, c, review)["total"] for c in CUTOFFS]
    return float(CUTOFFS[int(np.argmin(costs))])


def eur(x, dec=0):
    return f"€{x:,.{dec}f}"


def when(seconds):
    s = int(seconds)
    return f"Day {s // 86400 + 1}, {(s % 86400) // 3600:02d}:{(s % 3600) // 60:02d}"


def style(fig, height=340, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL, font=dict(family=FONT, color=INK, size=13),
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor=PANEL, bordercolor=LINE),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    )
    fig.update_xaxes(gridcolor="#EEF1EE", linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#EEF1EE", linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    return fig


CHART_CFG = {"displayModeBar": False}

# ---------------------------------------------------------------- hero
base = outcome(y, amt, prob, best_cutoff_from_past(5.0), 5.0)
saved_pct = 1 - base["total"] / NO_MODEL
by_eur_10 = test.nlargest(10, "exp_loss")
by_prob_10 = test.nlargest(10, "prob")
lift10 = by_eur_10.loc[by_eur_10.Class == 1, "Amount"].sum() / by_prob_10.loc[by_prob_10.Class == 1, "Amount"].sum()
prec_real = base["caught"] / (base["caught"] + base["fp_real"])

st.markdown(f"""
<p class="byline">Credit card fraud detection by Afeef Shaikh. <a href="{REPO_URL}" target="_blank">Code and notebooks on GitHub</a></p>
<h1 class="hero">On transactions it had never seen, the model cut fraud losses by {saved_pct:.0%}.</h1>
<p class="lede">The last {meta['dataset']['test_hours']:.1f} hours of card activity held {meta['dataset']['test_frauds']} frauds worth {eur(NO_MODEL)}.
The model was trained only on the hours before them, and its alert cutoff was picked by money, not accuracy.
About {round(prec_real * 4)} in 4 of its alerts were real fraud, even at real-world transaction volume.</p>
<div class="facts">
  <div><b>{eur(NO_MODEL - base['total'])}</b>kept, after paying €5 to review each alert</div>
  <div><b>{base['caught']} of {meta['dataset']['test_frauds']}</b>frauds caught, with {base['fp_real']:.0f} false alarms at real-world volume</div>
  <div><b>{lift10:.0f}× more</b>fraud money recovered from the first 10 reviews when alerts are sorted by € at risk</div>
</div>
""", unsafe_allow_html=True)

tab_cut, tab_queue, tab_why, tab_how = st.tabs(
    ["Set the cutoff", "Work the alert queue", "Why it was flagged", "How it was tested"])

# ---------------------------------------------------------------- tab 1: cutoff
with tab_cut:
    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        review = st.slider("Cost to review one alert", 1, 50, 5, format="€%d",
                           help="Analyst time per alert. A text message to the cardholder is cheap; a phone call is not.")
        mode = st.radio("Cutoff", ["Cheapest, chosen on past data", "Set it myself"], index=0,
                        help="The cheapest cutoff is found with walk-forward validation on earlier transactions, then applied to this later window.")
        auto_cut = best_cutoff_from_past(float(review))
        if mode == "Set it myself":
            cut = st.slider("Flag a transaction when its fraud score is at least", 0.01, 0.99, auto_cut, 0.01)
        else:
            cut = auto_cut
            st.markdown(f"<p class='note'>For €{review} per review, earlier data says flag anything scoring <b>{cut:.2f}</b> or higher.</p>",
                        unsafe_allow_html=True)
        st.markdown(f"<p class='note'>Fraud scores come from LightGBM. Missed fraud costs its full amount; "
                    f"false alarms are scaled ×{W:.2f} to undo the subsample.</p>", unsafe_allow_html=True)

    with right:
        o = outcome(y, amt, prob, cut, review)
        d = outcome(y, amt, prob, 0.50, review)
        saved = 1 - o["total"] / NO_MODEL
        verdict = ("costs more than having no model at all" if o["total"] > NO_MODEL
                   else f"is <b>{saved:.0%} cheaper</b> than having no model")
        st.markdown(f"""<p class="readout">At a cutoff of <b>{cut:.2f}</b>, the model catches <b>{o['caught']} frauds</b> worth
            {eur(o['caught_eur'])} and misses {o['missed']} worth {eur(o['missed_eur'])}. Analysts review about
            {o['alerts_real']:,.0f} alerts at real-world volume, costing {eur(o['review_eur'])}.
            Total cost is <b>{eur(o['total'])}</b>, which {verdict}.</p>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="counts">
            <div style="--c:{TEAL}"><b>{o['caught']}</b>frauds caught</div>
            <div style="--c:{CRIMSON}"><b>{o['missed']}</b>frauds missed</div>
            <div style="--c:{AMBER}"><b>{o['fp_real']:,.0f}</b>false alarms (real-world)</div>
            <div style="--c:{INK}"><b>{eur(o['total'])}</b>total cost</div></div>""", unsafe_allow_html=True)

        rows = ["No model", "Default 0.50", f"Yours {cut:.2f}"]
        missed_vals = [NO_MODEL, d["missed_eur"], o["missed_eur"]]
        review_vals = [0, d["review_eur"], o["review_eur"]]
        bar = go.Figure()
        bar.add_bar(y=rows, x=missed_vals, orientation="h", name="Missed fraud", marker_color=CRIMSON,
                    hovertemplate="%{y}<br>Missed fraud: €%{x:,.0f}<extra></extra>")
        bar.add_bar(y=rows, x=review_vals, orientation="h", name="Alert reviews", marker_color=AMBER,
                    hovertemplate="%{y}<br>Alert reviews: €%{x:,.0f}<extra></extra>")
        for r, m, v in zip(rows, missed_vals, review_vals):
            bar.add_annotation(y=r, x=m + v, text=f"  {eur(m + v)}", showarrow=False, xanchor="left",
                               font=dict(color=INK, size=13))
        bar.update_layout(barmode="stack", bargap=0.42, legend_traceorder="normal")
        bar.update_xaxes(tickprefix="€", separatethousands=True, range=[0, NO_MODEL * 1.32], nticks=5)
        bar.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(style(bar, height=230), width="stretch", config=CHART_CFG)

    curve = [outcome(y, amt, prob, c, review)["total"] for c in CUTOFFS]
    fig = go.Figure()
    fig.add_scatter(x=CUTOFFS, y=curve, mode="lines", line=dict(color=TEAL, width=2.5), name="Total cost",
                    hovertemplate="Cutoff %{x:.2f}<br>Total cost €%{y:,.0f}<extra></extra>")
    fig.add_hline(y=NO_MODEL, line=dict(color=GRAY, dash="dash", width=1.5))
    fig.add_annotation(x=0.99, y=NO_MODEL, text="No model", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(color=MUTED))
    fig.add_scatter(x=[cut], y=[o["total"]], mode="markers", marker=dict(size=12, color=TEAL, line=dict(color=PANEL, width=2)),
                    hovertemplate="Your cutoff %{x:.2f}<br>€%{y:,.0f}<extra></extra>", showlegend=False)
    fig.add_annotation(x=cut, y=o["total"], text=f"{cut:.2f}", showarrow=False, yshift=18, font=dict(color=INK))
    top = max(NO_MODEL * 1.25, min(max(curve), NO_MODEL * 2.2))
    fig.update_yaxes(range=[0, top], tickprefix="€", separatethousands=True, title=None)
    fig.update_xaxes(range=[0, 1], title="Cutoff: flag when fraud score is at least this")
    st.markdown("<h3 class='sec'>Every cutoff, priced</h3>", unsafe_allow_html=True)
    st.markdown("<p class='note'>Low cutoffs drown analysts in alerts; high cutoffs let fraud through. "
                "The flat valley means the model is forgiving: a wide range of cutoffs costs about the same.</p>",
                unsafe_allow_html=True)
    st.plotly_chart(style(fig, height=330, legend=False), width="stretch", config=CHART_CFG)

# ---------------------------------------------------------------- tab 2: queue
with tab_queue:
    budget = st.slider("Alerts your team can review in this window", 5, 150, 25, 5)
    by_eur = test.sort_values("exp_loss", ascending=False)
    by_prob = test.sort_values("prob", ascending=False)

    def eur_caught(order, n):
        top_n = order.head(n)
        return float(top_n.loc[top_n.Class == 1, "Amount"].sum())

    e_n, p_n = eur_caught(by_eur, budget), eur_caught(by_prob, budget)
    st.markdown(f"""<p class="readout">Sorted by <b>€ at risk</b> (fraud score × amount), {budget} reviews recover
        <b>{eur(e_n)}</b>, or {e_n / NO_MODEL:.0%} of all fraud money in the window. Sorted by fraud score alone,
        the same {budget} reviews recover {eur(p_n)}. The highest scores tend to be tiny card-testing charges.</p>""",
                unsafe_allow_html=True)

    ns = np.arange(1, 151)
    e_curve = by_eur.assign(f=lambda d: np.where(d.Class == 1, d.Amount, 0)).f.cumsum().to_numpy()[:150]
    p_curve = by_prob.assign(f=lambda d: np.where(d.Class == 1, d.Amount, 0)).f.cumsum().to_numpy()[:150]
    q = go.Figure()
    q.add_scatter(x=ns, y=e_curve, mode="lines", name="Sorted by € at risk", line=dict(color=TEAL, width=2.5),
                  hovertemplate="%{x} reviews<br>€%{y:,.0f} recovered<extra></extra>")
    q.add_scatter(x=ns, y=p_curve, mode="lines", name="Sorted by fraud score", line=dict(color=CRIMSON, width=2.5),
                  hovertemplate="%{x} reviews<br>€%{y:,.0f} recovered<extra></extra>")
    q.add_vline(x=budget, line=dict(color=INK, width=1, dash="dot"))
    q.add_hline(y=NO_MODEL, line=dict(color=GRAY, dash="dash", width=1.5))
    q.add_annotation(x=150, y=NO_MODEL, text="All fraud € in window", showarrow=False, xanchor="right",
                     yanchor="bottom", font=dict(color=MUTED))
    q.update_yaxes(tickprefix="€", separatethousands=True, range=[0, NO_MODEL * 1.08])
    q.update_xaxes(title="Alerts reviewed")
    st.plotly_chart(style(q, height=320), width="stretch", config=CHART_CFG)

    st.markdown(f"<h3 class='sec'>The queue, sorted by € at risk</h3>"
                f"<p class='note'>The {min(budget, 60)} alerts an analyst would open first in this window. "
                "Outcome is the bank's confirmed label, which the analyst would not see yet.</p>", unsafe_allow_html=True)
    rows_html = []
    for _, r in by_eur.head(min(budget, 60)).iterrows():
        fraud = r.Class == 1
        tag = "<span class='tag fraud'>Fraud</span>" if fraud else "<span class='tag legit'>Legit</span>"
        rows_html.append(f"<tr class='{'is-fraud' if fraud else ''}'><td>{when(r.Time)}</td><td>{tag}</td>"
                         f"<td>{eur(r.Amount, 2)}</td><td>{r.prob:.1%}</td><td>{eur(r.exp_loss, 2)}</td></tr>")
    st.markdown("<div class='ledger-wrap'><table class='ledger'><thead><tr><th>Time</th><th>Outcome</th>"
                "<th>Amount</th><th>Fraud score</th><th>€ at risk</th></tr></thead><tbody>"
                + "".join(rows_html) + "</tbody></table></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 3: why
with tab_why:
    feats = [c for c in shap_df.columns if c.startswith("shap_")]
    names = [c.replace("shap_", "") for c in feats]
    pool = shap_df.merge(test[["txn_id", "Time", "Class", "prob", "exp_loss"]], on="txn_id")
    pool = pool.sort_values("exp_loss", ascending=False).reset_index(drop=True)

    def label(r):
        kind = "fraud" if r.Class == 1 else "legit"
        return f"{eur(r.Amount, 2)} on {when(r.Time)}, score {r.prob:.1%} ({kind})"

    pick = st.selectbox("Transaction", range(len(pool)), format_func=lambda i: label(pool.iloc[i]),
                        help="Sorted by € at risk. Includes every fraud in the test window plus a sample of legitimate transactions.")
    r = pool.iloc[pick]
    contrib = pd.Series([r[f] for f in feats], index=names)
    values = pd.Series([r[n] for n in names], index=names)
    order = contrib.abs().sort_values(ascending=False).index
    top_k = list(order[:8])
    other = float(contrib.drop(top_k).sum())

    c1, c2 = st.columns([1, 1.9], gap="large")
    with c1:
        up = [f"{n} = {values[n]:.2f}" if n != "Amount" else f"Amount = {eur(values[n], 2)}"
              for n in order[:5] if contrib[n] > 0][:3]
        down = [f"{n} = {values[n]:.2f}" if n != "Amount" else f"Amount = {eur(values[n], 2)}"
                for n in order[:8] if contrib[n] < 0][:2]
        flagged = r.prob >= best_cutoff_from_past(5.0)
        st.markdown(f"""<div class="txn"><div class="amt">{eur(r.Amount, 2)}</div>
            <dl><dt>When</dt><dd>{when(r.Time)}</dd>
            <dt>Fraud score</dt><dd>{r.prob:.1%}</dd>
            <dt>€ at risk</dt><dd>{eur(r.exp_loss, 2)}</dd>
            <dt>Alert raised</dt><dd>{'Yes' if flagged else 'No'} (cutoff {best_cutoff_from_past(5.0):.2f})</dd>
            <dt>Confirmed outcome</dt><dd>{"<span class='tag fraud'>Fraud</span>" if r.Class == 1 else "<span class='tag legit'>Legit</span>"}</dd></dl></div>""",
                    unsafe_allow_html=True)
        st.markdown(f"<p class='readout' style='font-size:1rem'>Pushed toward fraud most by <b>{', '.join(up) if up else 'nothing strongly'}</b>"
                    + (f"; pulled back by {', '.join(down)}." if down else ".") + "</p>"
                    "<p class='note'>V1 to V28 are anonymized by the card issuer (PCA), so they can be ranked "
                    "but not named. In production these would be features like merchant type or distance from home.</p>",
                    unsafe_allow_html=True)
    with c2:
        steps = top_k[::-1]
        wf = go.Figure(go.Waterfall(
            orientation="h", base=meta["base_value"],
            y=["Other features"] + [f"{n} = {eur(values[n], 2) if n == 'Amount' else round(values[n], 2)}" for n in steps],
            x=[other] + [contrib[n] for n in steps],
            measure=["relative"] * (len(steps) + 1),
            increasing=dict(marker=dict(color=CRIMSON)), decreasing=dict(marker=dict(color=TEAL)),
            connector=dict(line=dict(color=LINE, width=1)),
            hovertemplate="%{y}<br>%{x:+.2f}<extra></extra>"))
        wf.update_xaxes(title="Fraud evidence (log-odds): left is safer, right is riskier")
        wf.update_yaxes(showgrid=False)
        st.plotly_chart(style(wf, height=400, legend=False), width="stretch", config=CHART_CFG)
        st.markdown(f"<p class='note'>Starts from the average transaction ({meta['base_value']:.1f}) and adds each feature's push. "
                    f"Red bars raise the fraud score; teal bars lower it. Final score: {r.prob:.1%}.</p>", unsafe_allow_html=True)

    imp = pd.Series(meta["global_importance"]).head(10)[::-1]
    gi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=INK,
                          hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    gi.update_xaxes(title="Average push on the fraud score, all test transactions")
    gi.update_yaxes(showgrid=False)
    st.markdown("<h3 class='sec'>What matters across all transactions</h3>", unsafe_allow_html=True)
    st.plotly_chart(style(gi, height=330, legend=False), width="stretch", config=CHART_CFG)

# ---------------------------------------------------------------- tab 4: how
with tab_how:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<h3 class='sec' style='margin-top:0.6rem'>Tested on the future, not a shuffle</h3>"
                    "<p class='note'>The tutorial this builds on shuffled transactions before splitting, so the model "
                    "trained on the future and was graded on the past. Trained on earlier hours and graded on later ones, "
                    "the same Random Forest catches noticeably less fraud.</p>", unsafe_allow_html=True)
        rs = go.Figure(go.Bar(x=["Random shuffle", "Past → future"], y=[0.811, 0.730], marker_color=[GRAY, TEAL],
                              text=["81% of fraud caught", "73% of fraud caught"], textposition="outside",
                              hovertemplate="%{x}: %{y:.1%}<extra></extra>", width=0.5))
        rs.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(style(rs, height=280, legend=False), width="stretch", config=CHART_CFG)
    with b:
        st.markdown("<h3 class='sec' style='margin-top:0.6rem'>Fraud doesn't sleep</h3>"
                    "<p class='note'>Legitimate spending dips overnight; fraud barely does. Between midnight and 5am a "
                    "transaction is about three times as likely to be fraud (2.68% vs 0.87%), so hour of day "
                    "replaced the raw timestamp as a feature.</p>", unsafe_allow_html=True)
        h = meta["hourly"]
        hr = go.Figure()
        hr.add_scatter(x=h["Hour"], y=h["legit_share"], name="Legitimate", mode="lines", line=dict(color=GRAY, width=2.5),
                       hovertemplate="%{x}:00 %{y:.1%} of legit<extra></extra>")
        hr.add_scatter(x=h["Hour"], y=h["fraud_share"], name="Fraud", mode="lines", line=dict(color=CRIMSON, width=2.5),
                       hovertemplate="%{x}:00 %{y:.1%} of fraud<extra></extra>")
        hr.update_yaxes(tickformat=".0%", title="Share of transactions")
        hr.update_xaxes(title="Hour of day", dtick=3)
        st.plotly_chart(style(hr, height=280), width="stretch", config=CHART_CFG)

    st.markdown("<h3 class='sec'>How the data was split</h3>"
                f"<p class='note'>Two days of European card transactions ({meta['dataset']['rows']:,} after removing duplicates; "
                f"{meta['dataset']['frauds']} frauds). The cutoff was chosen with walk-forward validation inside the first 80%: "
                "train on earlier chunks, predict the next, never peek ahead.</p>", unsafe_allow_html=True)
    sp = go.Figure()
    for name, x0, x1, col, txt in [("Train", 0, 33.4, "#C9D3CF", f"Train: {meta['dataset']['train_frauds']} frauds"),
                                   ("Validation", 33.4, 40.4, "#9DB2AB", f"Validate: {meta['dataset']['val_frauds']}"),
                                   ("Test", 40.4, 48, TEAL, f"Test: {meta['dataset']['test_frauds']}")]:
        sp.add_bar(y=["Hours"], x=[x1 - x0], base=[x0], orientation="h", marker_color=col, name=name,
                   text=[txt], textposition="inside", insidetextanchor="middle",
                   textfont=dict(color=PANEL if name == "Test" else INK),
                   hovertemplate=f"{name}: hours {x0}–{x1}<extra></extra>")
    sp.update_layout(barmode="overlay", bargap=0.2)
    sp.update_xaxes(range=[0, 48], dtick=6, title="Hours since the dataset began")
    sp.update_yaxes(showticklabels=False, showgrid=False)
    st.plotly_chart(style(sp, height=150, legend=False), width="stretch", config=CHART_CFG)

    c, dcol = st.columns(2, gap="large")
    with c:
        st.markdown("<h3 class='sec'>A default cutoff can lose money</h3>"
                    "<p class='note'>On the validation hours, Logistic Regression at the usual 0.50 cutoff raised so many "
                    "false alarms that it cost more than having no model. Its recall looked great (51 of 57).</p>",
                    unsafe_allow_html=True)
        models = ["LightGBM", "Random Forest", "Logistic Regression"]
        dv = go.Figure()
        dv.add_bar(x=models, y=[4775.15, 5037.88, 12651.79], name="At 0.50 cutoff", marker_color=CRIMSON,
                   hovertemplate="%{x} at 0.50: €%{y:,.0f}<extra></extra>")
        dv.add_bar(x=models, y=[4771.15, 4775.15, 5024.01], name="At cheapest cutoff", marker_color=TEAL,
                   hovertemplate="%{x} at cheapest: €%{y:,.0f}<extra></extra>")
        dv.add_hline(y=12231, line=dict(color=GRAY, dash="dash", width=1.5))
        dv.add_annotation(x=-0.45, y=12231, text="No model: €12,231", showarrow=False, xanchor="left",
                          yanchor="bottom", font=dict(color=MUTED))
        dv.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        dv.update_yaxes(tickprefix="€", separatethousands=True)
        st.plotly_chart(style(dv, height=300), width="stretch", config=CHART_CFG)
    with dcol:
        st.markdown("<h3 class='sec'>Models compared on validation</h3>"
                    "<p class='note'>Precision-recall area with legitimate transactions reweighted to real-world volume. "
                    "Random Forest and LightGBM tied on cost; LightGBM won the tie-break for its flatter cost curve, "
                    "identical results on any machine, and faster training.</p>", unsafe_allow_html=True)
        model_rows = [("Random Forest", "0.780", "42 of 57", "1"), ("LightGBM", "0.772", "43 of 57", "3"),
                      ("Logistic Regression", "0.746", "51 of 57", "357")]
        st.markdown("<div class='ledger-wrap'><table class='ledger plain'><thead><tr><th>Model</th><th>PR-AUC</th>"
                    "<th>Caught</th><th>False alarms</th></tr></thead><tbody>"
                    + "".join(f"<tr><td>{m}</td><td>{p}</td><td>{c}</td><td>{f}</td></tr>" for m, p, c, f in model_rows)
                    + "</tbody></table></div><p class='note' style='margin-top:0.5rem'>PR-AUC is real-world weighted; "
                    "caught and false alarms are at the 0.50 cutoff on 57 validation frauds.</p>",
                    unsafe_allow_html=True)

    st.markdown("<h3 class='sec'>Limits worth knowing</h3>"
                "<p class='note'>Two days of data, so the test window holds 74 frauds and a few transactions can move the "
                "numbers. The €5 review cost is an assumption, which is why the first tab lets you change it. "
                "Costs leave out customer friction and chargeback fees. The CSV keeps every fraud but only 50,000 of "
                "284,315 legitimate transactions; false alarms and precision are scaled back up to correct for that.</p>",
                unsafe_allow_html=True)

st.markdown(f"""<div class="footer">Built by Afeef Shaikh as a data science portfolio project.
    Model: LightGBM on the Kaggle ULB credit card dataset.
    <a href="{REPO_URL}" target="_blank">Notebooks and code</a> &nbsp;|&nbsp;
    <a href="https://afeefshaikh.com" target="_blank">afeefshaikh.com</a></div>""", unsafe_allow_html=True)
