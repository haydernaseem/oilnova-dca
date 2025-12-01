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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)
CORS(app)

# ======================
# 1) Decline Models
# ======================


def exp_decline(t, qi, Di):
    """Exponential decline: q = qi * exp(-Di * t)"""
    return qi * np.exp(-Di * t)


def harm_decline(t, qi, Di):
    """Harmonic decline: q = qi / (1 + Di * t)"""
    return qi / (1 + Di * t)


def hyp_decline(t, qi, Di, b):
    """Hyperbolic decline: q = qi / (1 + b * Di * t)^(1/b)"""
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
    """
    يحاول يتعامل مع:
    - عمود date → يحوله لأيام من أول تاريخ
    - auto-detect للأعمدة لو الأسماء مختلفة
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # معالجة الزمن
    if time_col not in df.columns:
        if "date" in cols_lower:
            dc = cols_lower["date"]
            df[dc] = pd.to_datetime(df[dc])
            df = df.sort_values(dc)
            df["t"] = (df[dc] - df[dc].iloc[0]).dt.days.astype(float)
            time_col = "t"
        else:
            time_col = df.columns[0]

    # معالجة الـ rate
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
    except Exception as e:
        print("Exponential fit failed:", e)

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
    except Exception as e:
        print("Harmonic fit failed:", e)

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
    except Exception as e:
        print("Hyperbolic fit failed:", e)

    if not results:
        raise ValueError("All DCA fits failed. Check data quality.")

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

        q_sim = np.clip(q_sim, 0, None)
        sims.append(q_sim)

    sims = np.vstack(sims)
    q_p10 = np.percentile(sims, 10, axis=0)
    q_p50 = np.percentile(sims, 50, axis=0)
    q_p90 = np.percentile(sims, 90, axis=0)

    return t, q_p10, q_p50, q_p90

# ======================
# 6) PDF Report Builder
# ======================


def create_pdf_report(models, best_model, eur, cutoff_rate, original_columns):
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(buffer.name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = f"DCA Analysis Report - {datetime.date.today().isoformat()}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(f"Best Model: <b>{best_model}</b>", styles["Normal"]))
    story.append(
        Paragraph(f"Cutoff Rate: <b>{cutoff_rate}</b>", styles["Normal"]))
    story.append(Paragraph(
        f"Estimated EUR: <b>{eur:,.2f}</b> (same units as rate * time)", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Original Columns:", styles["Heading3"]))
    story.append(Paragraph(", ".join(original_columns), styles["Normal"]))
    story.append(Spacer(1, 12))

    # Table of model parameters
    story.append(Paragraph("Model Fits Summary", styles["Heading2"]))
    table_data = [["Model", "qi", "Di", "b", "R²", "AIC", "RSS"]]

    for name, res in models.items():
        p = res["params"]
        qi = f"{p.get('qi', ''):,.4f}" if "qi" in p else ""
        Di = f"{p.get('Di', ''):,.6f}" if "Di" in p else ""
        b = f"{p.get('b', ''):,.4f}" if "b" in p else ""
        r2 = f"{res.get('r2', 0):.4f}"
        aic = f"{res.get('aic', 0):.2f}"
        rss = f"{res.get('rss', 0):.2f}"
        table_data.append([name, qi, Di, b, r2, aic, rss])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    story.append(table)

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Note: Decline curve analysis is based on historical production "
        "data and assumes stable operating conditions. Results are "
        "indicative and should be interpreted with engineering judgement.",
        styles["Italic"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ======================
# 7) API Endpoints
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
            return jsonify({"error": "Not enough valid data points after cleaning"}), 400

        models, best = run_dca(df)
        best_params = models[best]["params"]

        # History (cleaned)
        hist_t = df["t"].tolist()
        hist_q = df["q"].tolist()

        # Forecast + EUR
        t_f, q_f = base_forecast(best, best_params, days=days, dt=dt)
        eur = compute_eur(t_f, q_f, cutoff_rate=cutoff_rate)

        # Uncertainty
        t_u, q_p10, q_p50, q_p90 = monte_carlo_uncertainty(
            best, best_params, days=days, dt=dt, n_sim=200
        )

        return jsonify({
            "success": True,
            "best_model": best,
            "models": models,
            "history": {
                "t": hist_t,
                "q": hist_q
            },
            "forecast": {
                "t": t_f.tolist(),
                "q": q_f.tolist()
            },
            "uncertainty": {
                "t": t_u.tolist(),
                "q_p10": q_p10.tolist(),
                "q_p50": q_p50.tolist(),
                "q_p90": q_p90.tolist()
            },
            "eur": eur,
            "settings": {
                "cutoff_rate": cutoff_rate,
                "days": days,
                "dt": dt,
                "original_columns": list(df_raw.columns)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dca-report", methods=["POST"])
def dca_report():
    """يرجع PDF تقرير كامل."""
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
            return jsonify({"error": "Not enough valid data points after cleaning"}), 400

        models, best = run_dca(df)
        best_params = models[best]["params"]

        t_f, q_f = base_forecast(best, best_params, days=days, dt=dt)
        eur = compute_eur(t_f, q_f, cutoff_rate=cutoff_rate)

        pdf_buffer = create_pdf_report(
            models=models,
            best_model=best,
            eur=eur,
            cutoff_rate=cutoff_rate,
            original_columns=list(df_raw.columns)
        )

        filename = f"dca_report_{datetime.date.today().isoformat()}.pdf"
        return send_file(
            pdf_buffer.name,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# Run (local)
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

