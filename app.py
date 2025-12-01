import io
import tempfile
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from scipy.optimize import curve_fit

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Matplotlib for plots inside PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# ======================
# 1) Decline Models
# ======================

def exp_decline(t, qi, Di):
    return qi * np.exp(-Di * t)

def harm_decline(t, qi, Di):
    return qi / (1 + Di * t)

def hyp_decline(t, qi, Di, b):
    return qi / np.power(1 + b * Di * t, 1.0 / b)

# ======================
# 2) Helpers
# ======================

def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else 0.0

def compute_aic(n, rss, k):
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k

def clean_and_prepare_df(df, time_col="t", rate_col="q"):
    cols_lower = {c.lower(): c for c in df.columns}

    if time_col not in df.columns:
        if "date" in cols_lower:
            dc = cols_lower["date"]
            df[dc] = pd.to_datetime(df[dc])
            df = df.sort_values(dc)
            df["t"] = (df[dc] - df[dc].iloc[0]).dt.days.astype(float)
            time_col = "t"
        else:
            time_col = df.columns[0]

    if rate_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            if c != time_col:
                rate_col = c
                break

    df2 = df[[time_col, rate_col]].copy()
    df2 = df2.rename(columns={time_col: "t", rate_col: "q"})
    df2["t"] = pd.to_numeric(df2["t"], errors="coerce")
    df2["q"] = pd.to_numeric(df2["q"], errors="coerce")
    df2 = df2.dropna()
    df2 = df2[df2["q"] > 0]
    df2 = df2.sort_values("t")
    return df2.reset_index(drop=True)

# ======================
# 3) Main DCA Fit
# ======================

def run_dca(df):
    t = df["t"].values.astype(float)
    q = df["q"].values.astype(float)
    n = len(t)

    results = {}

    # Exponential
    try:
        popt, _ = curve_fit(
            exp_decline, t, q,
            p0=[q[0], 0.001],
            bounds=([0, 0], [np.inf, 5.0]),
            maxfev=8000
        )
        pred = exp_decline(t, *popt)
        rss = np.sum((q - pred) ** 2)
        results["exponential"] = {
            "params": {"qi": float(popt[0]), "Di": float(popt[1])},
            "rss": float(rss),
            "aic": compute_aic(n, rss, 2),
            "r2": r_squared(q, pred)
        }
    except Exception:
        pass

    # Harmonic
    try:
        popt, _ = curve_fit(
            harm_decline, t, q,
            p0=[q[0], 0.001],
            bounds=([0, 0], [np.inf, 5.0]),
            maxfev=8000
        )
        pred = harm_decline(t, *popt)
        rss = np.sum((q - pred) ** 2)
        results["harmonic"] = {
            "params": {"qi": float(popt[0]), "Di": float(popt[1])},
            "rss": float(rss),
            "aic": compute_aic(n, rss, 2),
            "r2": r_squared(q, pred)
        }
    except Exception:
        pass

    # Hyperbolic
    try:
        popt, _ = curve_fit(
            hyp_decline, t, q,
            p0=[q[0], 0.001, 0.5],
            bounds=([0, 0, 0], [np.inf, 1.0, 2.0]),
            maxfev=15000
        )
        pred = hyp_decline(t, *popt)
        rss = np.sum((q - pred) ** 2)
        results["hyperbolic"] = {
            "params": {
                "qi": float(popt[0]),
                "Di": float(popt[1]),
                "b": float(popt[2])
            },
            "rss": float(rss),
            "aic": compute_aic(n, rss, 3),
            "r2": r_squared(q, pred)
        }
    except Exception:
        pass

    best = sorted(results.items(), key=lambda x: x[1]["aic"])[0][0]
    return results, best

# ======================
# 4) Forecast + EUR
# ======================

def base_forecast(best_model, params, days=2000, dt=10):
    t = np.arange(0, days + dt, dt)
    if best_model == "exponential":
        q = exp_decline(t, params["qi"], params["Di"])
    elif best_model == "harmonic":
        q = harm_decline(t, params["qi"], params["Di"])
    else:
        q = hyp_decline(t, params["qi"], params["Di"], params["b"])
    q = np.clip(q, 0, None)
    return t, q

def compute_eur(t, q, cutoff_rate=5.0):
    above = np.where(q >= cutoff_rate)[0]
    if len(above) == 0:
        return 0.0
    idx_end = above[-1]
    t_use = t[:idx_end + 1]
    q_use = q[:idx_end + 1]
    eur = np.trapz(q_use, t_use)
    return float(eur)
# ======================
# 5) Monte Carlo Uncertainty
# ======================

def monte_carlo_uncertainty(best_model, params, days=2000, dt=10, n_sim=200):
    t = np.arange(0, days + dt, dt)
    sims = []
    rng = np.random.default_rng(42)

    sigma_factors = {"qi": 0.08, "Di": 0.2, "b": 0.2}

    for _ in range(n_sim):
        qi = max(params["qi"] * (1 + rng.normal(0, sigma_factors["qi"])), 1e-6)
        Di = max(params["Di"] * (1 + rng.normal(0, sigma_factors["Di"])), 1e-6)

        if best_model == "hyperbolic":
            b = params.get("b", 0.5)
            b_sim = max(b * (1 + rng.normal(0, sigma_factors["b"])), 0.01)
            q_sim = hyp_decline(t, qi, Di, b_sim)
        elif best_model == "harmonic":
            q_sim = harm_decline(t, qi, Di)
        else:
            q_sim = exp_decline(t, qi, Di)

        sims.append(np.clip(q_sim, 0, None))

    sims = np.vstack(sims)
    q_p10 = np.percentile(sims, 10, axis=0)
    q_p50 = np.percentile(sims, 50, axis=0)
    q_p90 = np.percentile(sims, 90, axis=0)

    return t, q_p10, q_p50, q_p90



# ======================
# 6) Plots for PDF
# ======================

def generate_plot(t_hist, q_hist, t_model, q_model, title):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.scatter(t_hist, q_hist, color="black", label="Historical", s=15)
    ax.plot(t_model, q_model, color="#1f77b4", linewidth=2.2, label="Model Fit")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.3)
    ax.legend()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(temp.name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return temp.name


# ======================
# 7) PDF Report Builder (FULL PROFESSIONAL)
# ======================

def create_pdf_report(models, best_model, eur, cutoff_rate, original_columns,
                       t_hist, q_hist, best_t, best_q):
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(buffer.name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # ----------------------------------------------------
    # (A) COVER PAGE — PROFESSIONAL DARK STYLE
    # ----------------------------------------------------
    title = Paragraph(
        "<para align='center'><font size=22><b>OILNOVA Decline Curve Report</b></font></para>",
        styles["Title"]
    )
    story.append(Spacer(1, 50))
    story.append(title)
    story.append(Spacer(1, 20))

    today = datetime.date.today().isoformat()
    meta = Paragraph(
        f"<para align='center'><font size=12 color='#555555'>Generated on {today}</font></para>",
        styles["Normal"]
    )
    story.append(meta)
    story.append(PageBreak())

    # ----------------------------------------------------
    # (B) EXECUTIVE SUMMARY
    # ----------------------------------------------------
    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"Best Model: <b>{best_model}</b>", styles["Normal"]))
    story.append(Paragraph(f"Cutoff Rate: <b>{cutoff_rate}</b>", styles["Normal"]))
    story.append(Paragraph(f"Estimated EUR: <b>{eur:,.2f}</b>", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Table of parameters
    story.append(Paragraph("<b>Model Parameters</b>", styles["Heading3"]))

    table_data = [["Model", "qi", "Di", "b", "R²", "AIC", "RSS"]]
    for name, res in models.items():
        p = res["params"]
        table_data.append([
            name,
            f"{p.get('qi', ''):,.4f}" if "qi" in p else "",
            f"{p.get('Di', ''):,.6f}" if "Di" in p else "",
            f"{p.get('b', ''):,.4f}" if "b" in p else "",
            f"{res.get('r2', 0):.4f}",
            f"{res.get('aic', 0):.2f}",
            f"{res.get('rss', 0):.2f}"
        ])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table)

    story.append(PageBreak())

    # ----------------------------------------------------
    # (C) THREE MODEL PLOTS — Exponential / Harmonic / Hyperbolic
    # ----------------------------------------------------

    exp_plot = None
    harm_plot = None
    hyp_plot = None

    if "exponential" in models:
        params = models["exponential"]["params"]
        t_m, q_m = base_forecast("exponential", params, 2000, 10)
        exp_plot = generate_plot(t_hist, q_hist, t_m, q_m, "Exponential Decline")

    if "harmonic" in models:
        params = models["harmonic"]["params"]
        t_m, q_m = base_forecast("harmonic", params, 2000, 10)
        harm_plot = generate_plot(t_hist, q_hist, t_m, q_m, "Harmonic Decline")

    if "hyperbolic" in models:
        params = models["hyperbolic"]["params"]
        t_m, q_m = base_forecast("hyperbolic", params, 2000, 10)
        hyp_plot = generate_plot(t_hist, q_hist, t_m, q_m, "Hyperbolic Decline")

    # Add plots vertically
    if exp_plot:
        story.append(Image(exp_plot, width=450, height=220))
        story.append(Spacer(1, 10))

    if harm_plot:
        story.append(Image(harm_plot, width=450, height=220))
        story.append(Spacer(1, 10))

    if hyp_plot:
        story.append(Image(hyp_plot, width=450, height=220))
        story.append(Spacer(1, 10))

    story.append(PageBreak())
    # ----------------------------------------------------
    # (D) FORECAST + UNCERTAINTY P10/P50/P90 PLOT
    # ----------------------------------------------------

    # Generate uncertainty ranges
    t_u, p10, p50, p90 = monte_carlo_uncertainty(best_model, models[best_model]["params"])

    # Create forecast plot
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.scatter(t_hist, q_hist, color="black", s=15, label="Historical")
    ax.plot(best_t, best_q, color="#1f77b4", linewidth=2, label="Best Model Forecast")
    ax.plot(t_u, p10, color="green", linestyle="--", label="P10")
    ax.plot(t_u, p50, color="darkgreen", linestyle="-.", label="P50")
    ax.plot(t_u, p90, color="limegreen", linestyle=":", label="P90")
    ax.set_title("Forecast & Uncertainty", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.3)
    ax.legend()

    forecast_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(forecast_temp.name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    story.append(Image(forecast_temp.name, width=460, height=260))
    story.append(PageBreak())

    # ----------------------------------------------------
    # (E) Final Notes Page
    # ----------------------------------------------------
    story.append(Paragraph("<b>Technical Note</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    note_text = (
        "Decline curve analysis is based on historical production data "
        "and assumes stable operating conditions. "
        "Results are indicative and should be interpreted with engineering judgement."
    )
    story.append(Paragraph(note_text, styles["Normal"]))
    story.append(Spacer(1, 20))

    # ----------------------------------------------------
    # FOOTER — POWERED BY OILNOVA AI
    # ----------------------------------------------------
    story.append(Spacer(1, 200))
    footer = Paragraph(
        "<para align='center'><font size=12 color='white'>"
        "<b>POWERED BY OILNOVA AI</b><br/>"
        "Iraq's First Artificial Intelligence Platform for Oil & Gas"
        "</font></para>",
        styles["Normal"]
    )

    # Black box
    story.append(
        Table(
            [[footer]],
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ],
            colWidths=[450]
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer



# ======================
# 8) API ENDPOINTS
# ======================

@app.route("/dca", methods=["POST"])
def dca_analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        cutoff_rate = float(request.form.get("cutoff_rate", 5.0))
        days = int(request.form.get("days", 2000))
        dt = float(request.form.get("dt", 30))
        time_col = request.form.get("time_col", "t")
        rate_col = request.form.get("rate_col", "q")

        raw = request.files["file"].read()
        df_raw = pd.read_csv(io.BytesIO(raw))

        if df_raw.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        df = clean_and_prepare_df(df_raw, time_col=time_col, rate_col=rate_col)
        if df.empty or len(df) < 5:
            return jsonify({"error": "Not enough valid data after cleaning"}), 400

        models, best = run_dca(df)
        best_params = models[best]["params"]

        # History
        t_hist = df["t"].tolist()
        q_hist = df["q"].tolist()

        # Forecast
        best_t, best_q = base_forecast(best, best_params, days=days, dt=dt)

        # EUR
        eur = compute_eur(best_t, best_q, cutoff_rate=cutoff_rate)

        # Uncertainty
        t_u, p10, p50, p90 = monte_carlo_uncertainty(best, best_params, days=days, dt=dt)

        return jsonify({
            "success": True,
            "best_model": best,
            "eur": eur,
            "models": models,
            "history": {"t": t_hist, "q": q_hist},
            "forecast": {"t": best_t.tolist(), "q": best_q.tolist()},
            "uncertainty": {
                "t": t_u.tolist(),
                "p10": p10.tolist(),
                "p50": p50.tolist(),
                "p90": p90.tolist()
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/dca-report", methods=["POST"])
def dca_report():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        cutoff_rate = float(request.form.get("cutoff_rate", 5.0))
        days = int(request.form.get("days", 2000))
        dt = float(request.form.get("dt", 30))
        time_col = request.form.get("time_col", "t")
        rate_col = request.form.get("rate_col", "q")

        raw = request.files["file"].read()
        df_raw = pd.read_csv(io.BytesIO(raw))

        df = clean_and_prepare_df(df_raw, time_col=time_col, rate_col=rate_col)
        if df.empty:
            return jsonify({"error": "Not enough valid data"}), 400

        models, best = run_dca(df)
        best_params = models[best]["params"]

        # History
        t_hist = df["t"].tolist()
        q_hist = df["q"].tolist()

        # Forecast
        best_t, best_q = base_forecast(best, best_params, days=days, dt=dt)

        # EUR
        eur = compute_eur(best_t, best_q, cutoff_rate)

        # Build PDF
        pdf_buffer = create_pdf_report(
            models=models,
            best_model=best,
            eur=eur,
            cutoff_rate=cutoff_rate,
            original_columns=list(df_raw.columns),
            t_hist=t_hist,
            q_hist=q_hist,
            best_t=best_t,
            best_q=best_q
        )

        filename = f"dca_report_{datetime.date.today().isoformat()}.pdf"
        return send_file(pdf_buffer.name, as_attachment=True,
                         download_name=filename, mimetype="application/pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ======================
# 9) RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
