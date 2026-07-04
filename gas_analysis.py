

# ============================== CELL 1 ===============================
import os, re, io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---- physical / experimental constants --------------------------------------
F_CONST     = 96485.33212     # Faraday constant (C/mol)
R_GAS       = 8.314462618     # ideal gas constant (J/mol/K)
T_AMBIENT_C = 21.2            # ambient temperature during measurements (deg C)
P_AMBIENT   = 1.013e5         # ambient pressure (Pa)
I_ON        = 0.05           # current threshold marking electrolysis onset (A)
V_READ_ERR  = 0.1            # syringe reading error on each gas volume (mL)
Z_H2, Z_O2  = 2, 4           # electrons transferred per H2 / O2 molecule

TK   = T_AMBIENT_C + 273.15
VMOL = R_GAS * TK / P_AMBIENT * 1e6        # molar gas volume (mL/mol) at T, P
# theoretical gas volume produced per coulomb of charge (mL/C):
ML_PER_C_H2 = VMOL / (Z_H2 * F_CONST)      # ~0.125 mL/C
ML_PER_C_O2 = VMOL / (Z_O2 * F_CONST)      # ~0.063 mL/C

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10,
    "axes.linewidth": 0.9, "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})
OUTDIR = "figures_gas"; os.makedirs(OUTDIR, exist_ok=True)
RUN_COLORS = None            # built in CELL 2 once the run list is known


# ========================= CELL 2 ============================
def have_gas():
    here = [f.lower() for f in os.listdir(".")]
    return any(re.search(r"gasrun\d+\.csv", f) for f in here) and \
           any("volume" in f and f.endswith(".csv") for f in here)

try:
    if not have_gas():
        from google.colab import files
        print("Select GasRun1.csv ... GasRun5.csv and "
              "data_gas_production_Volume_-_time.csv ...")
        files.upload()
except Exception as e:
    print("Not running in Colab or upload skipped:", e)

# electrical files: GasRun<n>.csv -> run label n
elec_map = {}
for f in os.listdir("."):
    m = re.search(r"gasrun(\d+)\.csv", f.lower())
    if m:
        elec_map[int(m.group(1))] = f
runs = sorted(elec_map)
print("Electrolysis runs found:", runs)
assert runs, "No GasRun*.csv files found - check the uploaded filenames."

# volume file: locate it by name
vol_file = next((f for f in os.listdir(".")
                 if "volume" in f.lower() and f.lower().endswith(".csv")), None)
assert vol_file, "Volume-time CSV not found."
print("Volume file:", vol_file)

RUN_COLORS = {n: c for n, c in zip(runs, plt.cm.viridis(np.linspace(0, 0.92, len(runs))))}


# ===================== CELL 3 =================================
def read_gasrun_csv(path):
    """Raspberry-Pi/INA219/ADS1115 electrolysis file.
    Returns columns t (s), V (V), I (A). Current is POSITIVE while the cell
    is driven as an electrolyser."""
    df = pd.read_csv(path).rename(columns={"Voltage": "V", "Current": "I", "Time": "t"})
    return df[["t", "V", "I"]].astype(float)

def read_volume_file(path):
    """Parse the hand-read volume file. It contains one block per run, each
    preceded by a 'Run, t_video_s, H2_ml, O2_ml' header and separated by blank
    lines. Returns {run: DataFrame(t_video, H2, O2)}."""
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = [p.strip() for p in line.replace("\r", "").split(",")]
            if len(parts) < 4 or not parts[0]:
                continue
            if parts[0].lower().startswith("run"):
                continue
            try:
                run = int(float(parts[0])); tv = float(parts[1])
                h = float(parts[2]); o = float(parts[3])
            except ValueError:
                continue
            out.setdefault(run, []).append((tv, h, o))
    return {r: pd.DataFrame(v, columns=["t_video", "H2", "O2"]).sort_values("t_video")
            for r, v in out.items()}

VOL = read_volume_file(vol_file)
print("Volume blocks found for runs:", sorted(VOL))


# ===================== CELL 4  - per-run charge + alignment ==================

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy >=2.0 or older

def analyse_run(n):
    df = read_gasrun_csv(elec_map[n])
    t, V, I = df["t"].values, df["V"].values, df["I"].values
    on = np.where(I > I_ON)[0]
    t0, t1 = t[on[0]], t[on[-1]]                       # electrolysis on-window
    vol = VOL[n].copy()
    tv = vol["t_video"].values

    # accumulated charge at each video timestamp
    Qcum = np.empty_like(tv, dtype=float)
    for k, tt in enumerate(tv):
        m = (t >= t0) & (t <= t0 + tt)
        Qcum[k] = trapz(I[m], t[m]) if m.sum() > 1 else 0.0
    vol["Q"] = Qcum
    Qtot = Qcum[-1]

    H2f, O2f = vol["H2"].iloc[-1], vol["O2"].iloc[-1]
    H2_th = ML_PER_C_H2 * Qtot                         # Faraday's law (z = 2)
    O2_th = ML_PER_C_O2 * Qtot                         # Faraday's law (z = 4)
    eta_H2 = H2f / H2_th
    eta_O2 = O2f / O2_th
    # reading-error propagation on the endpoint efficiency
    d_eta_H2 = eta_H2 * V_READ_ERR / H2f
    d_eta_O2 = eta_O2 * V_READ_ERR / O2f
    # volume-vs-charge regression (differential mL/C)
    aH, bH = np.polyfit(Qcum, vol["H2"].values, 1)
    aO, bO = np.polyfit(Qcum, vol["O2"].values, 1)

    return dict(run=n, t=t, V=V, I=I, t0=t0, t1=t1, vol=vol,
                Qtot=Qtot, H2f=H2f, O2f=O2f, H2_th=H2_th, O2_th=O2_th,
                eta_H2=eta_H2, eta_O2=eta_O2, d_eta_H2=d_eta_H2, d_eta_O2=d_eta_O2,
                slopeH=aH, slopeO=aO, ratio=H2f / O2f)

G = {n: analyse_run(n) for n in runs}

gas_summary = pd.DataFrame([{
    "run": d["run"], "Q (C)": d["Qtot"],
    "H2 meas (mL)": d["H2f"], "H2 theo (mL)": d["H2_th"], "eta_H2 (%)": d["eta_H2"]*100,
    "O2 meas (mL)": d["O2f"], "O2 theo (mL)": d["O2_th"], "eta_O2 (%)": d["eta_O2"]*100,
    "H2:O2": d["ratio"],
} for d in G.values()])
print(gas_summary.round(3).to_string(index=False))


# ===================== CELL 5 - repeatability statistics ====================
def round_up_unc(value, unc, sig=2):
    if unc == 0 or not np.isfinite(unc):
        return value, unc
    d = sig - 1 - int(math.floor(math.log10(abs(unc))))
    u = math.ceil(unc * 10**d) / 10**d
    return round(value, d), u

def repeat_stats(series, label, unit):
    x = np.asarray(series, float)
    m, s = x.mean(), x.std(ddof=1); cv = 100 * s / m
    v, u = round_up_unc(m, s)
    print(f"{label:10s} = {v} +/- {u} {unit:3s}  (CV = {cv:.1f} %)")
    return {"quantity": label, "mean": m, "sd": s, "CV (%)": cv, "unit": unit}

print("\nRun-to-run repeatability (n = %d):" % len(runs))
rep = [
    repeat_stats(gas_summary["Q (C)"],       "Q",      "C"),
    repeat_stats(gas_summary["eta_H2 (%)"],  "eta_H2", "%"),
    repeat_stats(gas_summary["eta_O2 (%)"],  "eta_O2", "%"),
    repeat_stats(gas_summary["H2:O2"],       "H2:O2",  ""),
]
rep_df = pd.DataFrame(rep)

# combined reading-error contribution (mean over runs) on the efficiencies
mean_dH2 = 100 * np.mean([d["d_eta_H2"] for d in G.values()])
mean_dO2 = 100 * np.mean([d["d_eta_O2"] for d in G.values()])
print(f"\nMean syringe-reading contribution: eta_H2 +/- {mean_dH2:.1f} %, "
      f"eta_O2 +/- {mean_dO2:.1f} %  (from +/-{V_READ_ERR} mL per reading)")


# ===================== CELL 6 - Figure 9: V(t) and I(t) =====================
# Representative run: cell voltage and drive current against time, with the electrolysis-on window shaded. Time is referenced to the onset (t_video).
REP_RUN = 1 if 1 in runs else runs[0]
d = G[REP_RUN]
trel = d["t"] - d["t0"]
fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
ax2 = ax1.twinx()
lV = ax1.plot(trel, d["V"], color="#1f4e79", lw=1.4, label="voltage")[0]
lI = ax2.plot(trel, d["I"]*1e3, color="#b5651d", lw=1.2, label="current")[0]
tv_end = d["vol"]["t_video"].iloc[-1]
ax1.axvspan(0, tv_end, color="#eaf0f6", zorder=0, label="electrolysis (video window)")
ax1.set_xlabel("Time from electrolysis onset (s)")
ax1.set_ylabel("Cell voltage (V)", color="#1f4e79")
ax2.set_ylabel("Drive current (mA)", color="#b5651d")
ax1.tick_params(axis="y", colors="#1f4e79"); ax2.tick_params(axis="y", colors="#b5651d")
ax1.set_xlim(-d["t0"], trel.max()); ax1.set_ylim(0, None); ax2.set_ylim(0, None)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.legend(handles=[lV, lI], frameon=False, loc="center right")
fig.savefig(f"{OUTDIR}/figI_electrolysis_timeseries.pdf")
fig.savefig(f"{OUTDIR}/figI_electrolysis_timeseries.png")
plt.show()


# ============== CELL 7 - Figure 10: gas volume vs accumulated charge =========
# Measured H2 and O2 volume against charge passed, for all runs, with the theoretical Faraday-law lines.
fig, ax = plt.subplots(figsize=(7.0, 4.8))
for n in runs:
    v = G[n]["vol"]
    ax.errorbar(v["Q"], v["H2"], yerr=V_READ_ERR, fmt="o", ms=3.5, capsize=2,
                color=RUN_COLORS[n], alpha=0.9, lw=0.8,
                label=f"Run {n}")
    ax.errorbar(v["Q"], v["O2"], yerr=V_READ_ERR, fmt="s", ms=3.5, capsize=2,
                color=RUN_COLORS[n], alpha=0.9, lw=0.8, mfc="white")
Qline = np.array([0, max(d["Qtot"] for d in G.values())])
ax.plot(Qline, ML_PER_C_H2*Qline, "k--", lw=1.6, label=r"Faraday H$_2$ ($z=2$)")
ax.plot(Qline, ML_PER_C_O2*Qline, "k:",  lw=1.6, label=r"Faraday O$_2$ ($z=4$)")
ax.text(70, ML_PER_C_H2*70 + 1.3, r"H$_2$ (circles)", fontsize=10, color="0.3",
        ha="center", rotation=0,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
ax.text(95, ML_PER_C_O2*95 + 1.1, r"O$_2$ (squares)", fontsize=10, color="0.3",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
ax.set_xlabel("Charge passed (C)"); ax.set_ylabel("Collected gas volume (mL)")
ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=9)
fig.savefig(f"{OUTDIR}/figJ_volume_vs_charge.pdf")
fig.savefig(f"{OUTDIR}/figJ_volume_vs_charge.png")
plt.show()


# ============== CELL 8 - Figure 11: Faradaic efficiency per run ==============
# Endpoint Faradaic efficiency for H2 and O2 per run, with syringe-reading error bars, and the run-mean drawn as a horizontal band.
fig, ax = plt.subplots(figsize=(7.0, 4.3))
x = np.array(runs, float)
eH = gas_summary["eta_H2 (%)"].values; eO = gas_summary["eta_O2 (%)"].values
dH = 100*np.array([d["d_eta_H2"] for d in G.values()])
dO = 100*np.array([d["d_eta_O2"] for d in G.values()])
ax.errorbar(x-0.06, eH, yerr=dH, fmt="o", ms=6, capsize=3, color="#1f4e79", label=r"H$_2$")
ax.errorbar(x+0.06, eO, yerr=dO, fmt="s", ms=6, capsize=3, color="#b5651d", label=r"O$_2$")
for val, col in [(eH.mean(), "#1f4e79"), (eO.mean(), "#b5651d")]:
    ax.axhline(val, color=col, lw=1, ls="--", alpha=0.6)
ax.axhline(100, color="0.4", lw=1, ls=":")
ax.text(x[-1]+0.15, 100, "ideal", fontsize=9, color="0.4", va="center")
ax.set_xlabel("Run"); ax.set_ylabel("Faradaic efficiency (%)")
ax.set_xticks(runs); ax.set_ylim(80, 104)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(frameon=False, loc="lower right")
fig.savefig(f"{OUTDIR}/figK_faradaic_efficiency.pdf")
fig.savefig(f"{OUTDIR}/figK_faradaic_efficiency.png")
plt.show()


# ===================== CELL 9 ====================
gas_summary.round(3).to_csv(f"{OUTDIR}/table_gas_faradaic.csv", index=False)
rep_df.round(4).to_csv(f"{OUTDIR}/table_gas_repeatability.csv", index=False)

# tidy per-(run, timestamp) volume-charge table, handy for the appendix
vc = pd.concat([G[n]["vol"].assign(run=n) for n in runs], ignore_index=True)
vc = vc[["run", "t_video", "Q", "H2", "O2"]]
vc.round(3).to_csv(f"{OUTDIR}/table_volume_charge.csv", index=False)
print(gas_summary.round(3).to_string(index=False))

try:
    import shutil
    from google.colab import files
    shutil.make_archive("gas_results", "zip", OUTDIR)
    files.download("gas_results.zip")
except Exception as e:
    print("Saved to", OUTDIR, "(download step skipped outside Colab):", e)
