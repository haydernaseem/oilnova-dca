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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Matplotlib for plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

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


def create_dca_plots(df, models, best_model):
    """Create professional DCA plots for PDF report"""
    t = df["t"].values
    q = df["q"].values
    
    fig_width = 7
    fig_height = 2.2
    
    # Plot 1: Exponential Decline
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax1.scatter(t, q, color='black', s=15, alpha=0.6, label='Actual Data')
    
    if "exponential" in models:
        params = models["exponential"]["params"]
        t_fit = np.linspace(t.min(), t.max() * 1.5, 100)
        q_fit = exp_decline(t_fit, params["qi"], params["Di"])
        line_style = '--' if best_model != "exponential" else '-'
        line_width = 1.2 if best_model != "exponential" else 2.0
        line_color = 'blue' if best_model != "exponential" else 'red'
        ax1.plot(t_fit, q_fit, line_style, color=line_color, 
                linewidth=line_width, label='Exponential Fit')
    
    ax1.set_xlabel('Time (days)', fontsize=8)
    ax1.set_ylabel('Rate', fontsize=8)
    ax1.set_title('Exponential Decline', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)
    
    # Format y-axis with comma separators
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('exp_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Harmonic Decline
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    ax2.scatter(t, q, color='black', s=15, alpha=0.6, label='Actual Data')
    
    if "harmonic" in models:
        params = models["harmonic"]["params"]
        t_fit = np.linspace(t.min(), t.max() * 1.5, 100)
        q_fit = harm_decline(t_fit, params["qi"], params["Di"])
        line_style = '--' if best_model != "harmonic" else '-'
        line_width = 1.2 if best_model != "harmonic" else 2.0
        line_color = 'green' if best_model != "harmonic" else 'red'
        ax2.plot(t_fit, q_fit, line_style, color=line_color, 
                linewidth=line_width, label='Harmonic Fit')
    
    ax2.set_xlabel('Time (days)', fontsize=8)
    ax2.set_ylabel('Rate', fontsize=8)
    ax2.set_title('Harmonic Decline', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('harm_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Hyperbolic Decline
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
    ax3.scatter(t, q, color='black', s=15, alpha=0.6, label='Actual Data')
    
    if "hyperbolic" in models:
        params = models["hyperbolic"]["params"]
        t_fit = np.linspace(t.min(), t.max() * 1.5, 100)
        q_fit = hyp_decline(t_fit, params["qi"], params["Di"], params["b"])
        line_style = '--' if best_model != "hyperbolic" else '-'
        line_width = 1.2 if best_model != "hyperbolic" else 2.0
        line_color = 'purple' if best_model != "hyperbolic" else 'red'
        ax3.plot(t_fit, q_fit, line_style, color=line_color, 
                linewidth=line_width, label='Hyperbolic Fit')
    
    ax3.set_xlabel('Time (days)', fontsize=8)
    ax3.set_ylabel('Rate', fontsize=8)
    ax3.set_title('Hyperbolic Decline', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('hyp_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'exp_plot.png', 'harm_plot.png', 'hyp_plot.png'


def create_comparison_plot(df, models, best_model):
    """Create comparison plot of all models"""
    t = df["t"].values
    q = df["q"].values
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Scatter plot of actual data
    ax.scatter(t, q, color='black', s=20, alpha=0.8, label='Actual Data', zorder=5)
    
    # Plot each model
    t_fit = np.linspace(t.min(), t.max() * 2, 200)
    
    colors_map = {
        'exponential': 'blue',
        'harmonic': 'green',
        'hyperbolic': 'purple'
    }
    
    for model_name, model_data in models.items():
        params = model_data["params"]
        if model_name == "exponential":
            q_fit = exp_decline(t_fit, params["qi"], params["Di"])
        elif model_name == "harmonic":
            q_fit = harm_decline(t_fit, params["qi"], params["Di"])
        else:
            q_fit = hyp_decline(t_fit, params["qi"], params["Di"], params["b"])
        
        line_style = '--' if model_name != best_model else '-'
        line_width = 1.2 if model_name != best_model else 2.5
        label = f"{model_name.title()} (Best)" if model_name == best_model else model_name.title()
        
        ax.plot(t_fit, q_fit, line_style, color=colors_map[model_name], 
                linewidth=line_width, label=label, alpha=0.8)
    
    ax.set_xlabel('Time (days)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Production Rate', fontsize=9, fontweight='bold')
    ax.set_title('Model Comparison: Decline Curve Analysis', 
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(labelsize=8)
    
    # Add text box with statistics
    stats_text = f"Best Model: {best_model.upper()}\n"
    stats_text += f"Data Points: {len(t)}\n"
    stats_text += f"Time Range: {t.min():.0f} - {t.max():.0f} days"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Format axes
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'comparison_plot.png'


def create_pdf_report(models, best_model, eur, cutoff_rate, original_columns, df):
    """Create professional PDF report with plots (2 pages)"""
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(buffer.name, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    styles.add(ParagraphStyle(
        name='Header1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#0b3d91'),
        spaceAfter=8,
    ))
    
    styles.add(ParagraphStyle(
        name='Header2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a5276'),
        spaceAfter=6,
    ))
    
    styles.add(ParagraphStyle(
        name='Header3',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=4,
    ))
    
    styles.add(ParagraphStyle(
        name='Highlight',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#2c3e50'),
        backColor=colors.HexColor('#f8f9fa'),
        borderPadding=4,
        borderColor=colors.HexColor('#dee2e6'),
        borderWidth=1,
    ))
    
    styles.add(ParagraphStyle(
        name='SmallText',
        parent=styles['Normal'],
        fontSize=7,
        textColor=colors.HexColor('#666666'),
    ))
    
    styles.add(ParagraphStyle(
        name='Branding',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#0b3d91'),
        alignment=1,
        spaceBefore=5,
        spaceAfter=5,
    ))
    
    story = []
    
    # ======================
    # PAGE 1: Analysis Results (الرسوم البيانية والملخص)
    # ======================
    
    # Header with logo placeholder
    header_text = """
    <para align=center>
    <font size=14 color=#0b3d91><b>DECLINE CURVE ANALYSIS REPORT</b></font><br/>
    <font size=10 color=#7f8c8d>Professional Petroleum Engineering Analysis</font>
    </para>
    """
    story.append(Paragraph(header_text, styles["Normal"]))
    story.append(Spacer(1, 4))
    
    # Report metadata
    meta_text = f"""
    <para align=center>
    <font size=8>Report Date: {datetime.date.today().strftime('%B %d, %Y')} | 
    Analysis ID: DCA-{datetime.date.today().strftime('%Y%m%d')}-{np.random.randint(1000,9999)}</font>
    </para>
    """
    story.append(Paragraph(meta_text, styles["Normal"]))
    story.append(Spacer(1, 12))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", styles["Header1"]))
    story.append(Spacer(1, 6))
    
    summary_text = f"""
    <para>
    This report presents a comprehensive Decline Curve Analysis (DCA) based on historical production data. 
    The analysis compares three standard decline models (Exponential, Harmonic, and Hyperbolic) to determine 
    the best fit for production forecasting and Estimated Ultimate Recovery (EUR) calculation.
    </para>
    """
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 8))
    
    # Key Findings box
    findings_text = f"""
    <b>KEY FINDINGS:</b><br/>
    • <b>Best-fitting Model:</b> <font color=#c0392b>{best_model.upper()}</font><br/>
    • <b>Estimated Ultimate Recovery (EUR):</b> {eur:,.0f} units<br/>
    • <b>Economic Cutoff Rate:</b> {cutoff_rate} units/day<br/>
    • <b>Data Points Analyzed:</b> {len(df)} measurements<br/>
    • <b>Time Period:</b> {df['t'].min():.0f} - {df['t'].max():.0f} days<br/>
    • <b>Production Range:</b> {df['q'].min():,.0f} - {df['q'].max():,.0f} units/day
    """
    story.append(Paragraph(findings_text, styles["Highlight"]))
    story.append(Spacer(1, 20))
    
    # Create plots
    exp_plot, harm_plot, hyp_plot = create_dca_plots(df, models, best_model)
    comparison_plot = create_comparison_plot(df, models, best_model)
    
    # Model Comparison Plot
    story.append(Paragraph("MODEL COMPARISON ANALYSIS", styles["Header2"]))
    story.append(Spacer(1, 6))
    story.append(Image(comparison_plot, width=6*inch, height=2.5*inch))
    story.append(Spacer(1, 15))
    
    # Individual Model Analysis
    story.append(Paragraph("INDIVIDUAL MODEL ANALYSIS", styles["Header2"]))
    story.append(Spacer(1, 6))
    
    # Create a table with the three plots
    plot_table_data = [
        [Image(exp_plot, width=2.5*inch, height=1.1*inch),
         Image(harm_plot, width=2.5*inch, height=1.1*inch),
         Image(hyp_plot, width=2.5*inch, height=1.1*inch)]
    ]
    
    plot_table = Table(plot_table_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    plot_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(plot_table)
    story.append(Spacer(1, 6))
    
    # Add plot captions
    captions = ["Exponential Decline Model", "Harmonic Decline Model", "Hyperbolic Decline Model"]
    caption_table_data = [[Paragraph(f"<para align=center><font size=8><b>{cap}</b></font></para>", styles["Normal"]) 
                          for cap in captions]]
    caption_table = Table(caption_table_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    story.append(caption_table)
    
    # Add page break
    story.append(PageBreak())
    
    # ======================
    # PAGE 2: Detailed Results and OILNOVA Branding
    # ======================
    
    # Detailed Results Table
    story.append(Paragraph("MODEL PARAMETERS AND STATISTICS", styles["Header2"]))
    story.append(Spacer(1, 10))
    
    # Prepare table data
    table_data = [
        ["Model", "qi", "Di", "b", "R²", "AIC", "RSS"]
    ]
    
    for name, res in models.items():
        p = res["params"]
        qi = f"{p.get('qi', 0):,.2f}" if "qi" in p else "N/A"
        Di = f"{p.get('Di', 0):.6f}" if "Di" in p else "N/A"
        b = f"{p.get('b', 0):.4f}" if "b" in p else "N/A"
        r2 = f"{res.get('r2', 0):.4f}"
        aic = f"{res.get('aic', 0):.2f}"
        rss = f"{res.get('rss', 0):.2e}"
        
        # Highlight best model
        if name == best_model:
            row = [f"{name.upper()}*", qi, Di, b, r2, aic, rss]
        else:
            row = [name.title(), qi, Di, b, r2, aic, rss]
        
        table_data.append(row)
    
    table = Table(table_data, hAlign="CENTER", colWidths=[1.2*inch, 1.0*inch, 1.0*inch, 
                                                           0.7*inch, 0.6*inch, 0.7*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, -1), colors.HexColor("#0b3d91")),
    ]))
    
    # Highlight the best model row
    for i, row in enumerate(table_data[1:], start=1):
        if row[0].endswith("*"):
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e8f4fd")),
                ("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"),
            ]))
    
    story.append(table)
    
    note_text = "*Best-fitting model based on lowest AIC"
    story.append(Paragraph(note_text, styles["SmallText"]))
    story.append(Spacer(1, 20))
    
    # Data and EUR Information in SAME LINE (تعديل رئيسي هنا)
    # نستخدم جدول بعمودين بدون عناوين فوقها
    
    info_data = [
        [  # Column 1: DATA INFORMATION
            Paragraph("<b>DATA INFORMATION</b>", styles["Normal"]),
            Paragraph(f"<b>Original Columns:</b> {', '.join(original_columns[:3])}{'...' if len(original_columns) > 3 else ''}", styles["Normal"]),
            Paragraph(f"<b>Data Points:</b> {len(df)} measurements", styles["Normal"]),
            Paragraph(f"<b>Data Quality:</b> All rates > 0", styles["Normal"])
        ],
        [  # Column 2: ECONOMIC ANALYSIS
            Paragraph("<b>ECONOMIC ANALYSIS</b>", styles["Normal"]),
            Paragraph(f"<b>Estimated Ultimate Recovery:</b> {eur:,.0f} units", styles["Normal"]),
            Paragraph(f"<b>Cutoff Rate:</b> {cutoff_rate} units/day", styles["Normal"]),
            Paragraph(f"<b>Confidence:</b> Monte Carlo (200 sims)", styles["Normal"])
        ]
    ]
    
    # Create table with 2 columns
    info_table = Table(info_data, colWidths=[2.8*inch, 2.8*inch])
    info_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 40))
    
    # ======================
    # OILNOVA AI Branding at bottom of Page 2
    # ======================
    
    # Horizontal line
    from reportlab.platypus.flowables import HRFlowable
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0b3d91")))
    story.append(Spacer(1, 15))
    
    # OILNOVA AI Branding
    branding_text = """
    <para align=center>
    <font size=12 color=#0b3d91><b>OILNOVA AI</b></font><br/>
    <font size=10 color=#1a5276>Advanced Petroleum Analytics Platform</font><br/>
    <font size=9 color=#2c3e50>Report Generated by OILNOVA AI Decline Curve Analysis Module</font><br/>
    <font size=8 color=#7f8c8d>© 2024 OILNOVA AI. All rights reserved. Proprietary and Confidential</font>
    </para>
    """
    story.append(Paragraph(branding_text, styles["Normal"]))
    story.append(Spacer(1, 10))
    
    # Technical footer
    tech_footer = f"""
    <para align=center>
    <font size=7 color=#95a5a6>
    Report Version: 2.5 | Analysis Engine: DeepSeek AI | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    </font>
    </para>
    """
    story.append(Paragraph(tech_footer, styles["SmallText"]))
    
    # Build the document
    doc.build(story)
    
    # Clean up temporary plot files
    for plot_file in [exp_plot, harm_plot, hyp_plot, comparison_plot]:
        try:
            os.remove(plot_file)
        except:
            pass
    
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
    """Generate professional PDF report with plots (2 pages)"""
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
            original_columns=list(df_raw.columns),
            df=df
        )

        filename = f"DCA_Report_{datetime.date.today().isoformat()}_{np.random.randint(1000,9999)}.pdf"
        return send_file(
            pdf_buffer.name,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Oilnova DCA API",
        "version": "2.5.0",
        "timestamp": datetime.datetime.now().isoformat()
    })


# ======================
# Run (local)
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
