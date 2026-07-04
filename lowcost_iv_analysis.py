# ============================== CELL 1 ===============================
import os, re, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

T_AMBIENT_C  = 21.2          # ambient temperature during measurements (deg C)
P_AMBIENT    = 1.013e5       # ambient pressure used for gas conversion (Pa)
R_GAS        = 8.314462618   # ideal gas constant (J/mol/K)
OCV_WINDOW   = 30.0          # OCV averaging window (s) -- matches Sec. 3.1
I_ON         = 1.5e-3        # current threshold to detect the discharge phase (A)
OVERLAP_IMAX = 0.150         # upper current (A) of the potentiostat-comparable window
BIN_DI       = 3e-3          # current-bin width for the polarisation curves (A)
CMP_CURRENTS = np.array([20, 50, 100, 130]) * 1e-3   # currents for the V comparison

CSV2RUN = {5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10,
    "axes.linewidth": 0.9, "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})
OUTDIR = "figures_lowcost"; os.makedirs(OUTDIR, exist_ok=True)
RUN_COLORS = None


# ========================= CELL 2 ============================
def have_lowcost():
    return any(re.search(r"flooriv\d+\.csv", f.lower()) for f in os.listdir("."))

try:
    if not have_lowcost():
        from google.colab import files
        print("Select the low-cost CSV files (FloorIV5.csv ... FloorIV12.csv), "
              "and optionally the Gamry .DTA reference files ...")
        files.upload()
except Exception as e:
    print("Not running in Colab or upload skipped:", e)

# low-cost CSV files -> run labels
lc_map = {}
for f in os.listdir("."):
    m = re.search(r"flooriv(\d+)\.csv", f.lower())
    if m and int(m.group(1)) in CSV2RUN:
        lc_map[CSV2RUN[int(m.group(1))]] = f
runs = sorted(lc_map)
print("Low-cost runs found:", runs)
assert runs, "No FloorIV*.csv files found - check the uploaded filenames."

giv_map, gocv_map = {}, {}
for f in os.listdir("."):
    if not f.lower().endswith(".dta"):
        continue
    m_iv  = re.search(r"_IV_(\d+)", f, re.IGNORECASE)
    m_ocv = re.search(r"ocv(\d+)",  f, re.IGNORECASE)
    if m_iv:   giv_map[int(m_iv.group(1))] = f
    elif m_ocv: gocv_map[int(m_ocv.group(1))] = f
gamry_runs = sorted(set(giv_map) & set(gocv_map))
HAVE_GAMRY = len(gamry_runs) > 0
print("Gamry reference runs found:" if HAVE_GAMRY
      else "No Gamry .DTA uploaded - using published Section 4.1 summary as reference.",
      gamry_runs)

RUN_COLORS = {n: c for n, c in zip(runs, plt.cm.viridis(np.linspace(0, 0.92, len(runs))))}


# ===================== CELL 3 =================================
def read_lowcost_csv(path):
    """Raspberry-Pi/INA219/ADS1115 .csv -> columns t (s), V (V), I (A).
    The low-cost current is POSITIVE while the cell discharges."""
    df = pd.read_csv(path).rename(columns={"Voltage": "V", "Current": "I", "Time": "t"})
    return df[["t", "V", "I"]].astype(float)

def read_gamry_dta(path):
    """Gamry .DTA (latin-1, comma decimals) -> t (s), Vf (V), Im (A).
    Im is negative while the cell discharges."""
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("CURVE") and "\tTABLE" in l)
    header = [h.strip() for h in lines[start + 1].split("\t")]
    col = {name: header.index(name) for name in ("T", "Vf", "Im")}
    t, vf, im = [], [], []
    for l in lines[start + 3:]:
        if not l.strip():
            continue
        p = l.split("\t")
        try:
            t.append(float(p[col["T"]].replace(",", ".")))
            vf.append(float(p[col["Vf"]].replace(",", ".")))
            im.append(float(p[col["Im"]].replace(",", ".")))
        except (IndexError, ValueError):
            continue
    return pd.DataFrame({"t": t, "Vf": vf, "Im": im})


# ===================== CELL 4 - low-cost per-run analysis ====================
def analyse_lowcost(run):
    df = read_lowcost_csv(lc_map[run])
    t, V, I = df["t"].values, df["V"].values, df["I"].values
    P = V * I
    on = np.where(I > I_ON)[0]
    i0, i1 = on[0], on[-1]                                  # discharge window
    pre = (np.arange(len(t)) < i0) & (np.abs(I) < 1e-3)
    Voc = V[pre].mean() if pre.any() else V[t < OCV_WINDOW].mean()
    td, Vd, Id, Pd = t[i0:i1+1], V[i0:i1+1], I[i0:i1+1], P[i0:i1+1]

    # --- binned polarisation curve over the rising portion (up to peak power) -
    kg = int(np.argmax(Pd))
    Ir, Vr = Id[:kg+1], Vd[:kg+1]
    edges = np.arange(0, Ir.max() + BIN_DI, BIN_DI)
    idx = np.digitize(Ir, edges)
    Ib, Vb = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() >= 3:
            Ib.append(Ir[m].mean()); Vb.append(np.median(Vr[m]))
    Ib = np.r_[0.0, Ib]; Vb = np.r_[Voc, Vb]               # anchor at (0, OCV)
    Pb = Ib * Vb
    kb = int(np.argmax(Pb))                                # apparent (transient) peak

    Vat = lambda ic: (np.interp(ic, Ib, Vb) if ic <= Ib.max() else np.nan)
    return dict(run=run, t=t, V=V, I=I, P=P, i0=i0, i1=i1, td=td, Vd=Vd, Id=Id, Pd=Pd,
                Voc=Voc, Ib=Ib, Vb=Vb, Pb=Pb, dur=t.max(),
                Imax=Ib.max(), Papp=Pb[kb], Vapp=Vb[kb], Iapp=Ib[kb],
                V100=Vat(0.100), V150=Vat(0.150))

LC = {n: analyse_lowcost(n) for n in runs}

lc_summary = pd.DataFrame([{
    "run": d["run"], "OCV (V)": d["Voc"], "Imax (mA)": d["Imax"]*1e3,
    "V@100mA (V)": d["V100"], "Papp (mW)": d["Papp"]*1e3, "Iapp (mA)": d["Iapp"]*1e3,
} for d in LC.values()])
print("Per-run low-cost results (apparent peak = transient, see text):")
print(lc_summary.round(3).to_string(index=False))

Igrid_lc = np.linspace(0.0, OVERLAP_IMAX, 300)
Vlc = np.array([np.interp(Igrid_lc, d["Ib"], d["Vb"]) for d in LC.values()])
Vlc_mean, Vlc_std = Vlc.mean(0), Vlc.std(0, ddof=1)
Plc_mean = Vlc_mean * Igrid_lc


# ============= CELL 5 - potentiostat reference ==================
if HAVE_GAMRY:
    g_curves = []
    for n in gamry_runs:
        df = read_gamry_dta(giv_map[n])
        Ig = -df["Im"].values
        Vg = df["Vf"].values
        g_curves.append((Ig, Vg))
    Imax_common = min(Ig.max() for Ig, _ in g_curves)
    Igrid_g = np.linspace(0.0, 0.99 * Imax_common, 400)
    Vg_int = []
    for Ig, Vg in g_curves:
        o = np.argsort(Ig)
        Vg_int.append(np.interp(Igrid_g, Ig[o], Vg[o]))
    Vg_int = np.array(Vg_int)
    Vg_mean, Vg_std = Vg_int.mean(0), Vg_int.std(0, ddof=1)
    Pg_mean = Vg_mean * Igrid_g
    kg_mean = int(np.argmax(Pg_mean))
    GAMRY_MPP = (Pg_mean[kg_mean], Vg_mean[kg_mean], Igrid_g[kg_mean])
    print(f"Gamry mean curve: Pmax = {GAMRY_MPP[0]*1e3:.0f} mW "
          f"at V = {GAMRY_MPP[1]:.3f} V, I = {GAMRY_MPP[2]*1e3:.0f} mA")
else:
    # Published Section 4.1 mean-curve anchors (plotting/labelling only).
    Igrid_g = np.array([0, 20, 50, 75, 100, 122, 147]) * 1e-3
    Vg_mean = np.array([1.00, 0.80, 0.77, 0.74, 0.72, 0.68, 0.63])
    Vg_std  = np.full_like(Vg_mean, 0.03)
    Pg_mean = Vg_mean * Igrid_g
    GAMRY_MPP = (0.083, 0.68, 0.122)


# ============= CELL 6 - cross-method comparison ==============================
Icmp = np.linspace(0.0, min(OVERLAP_IMAX, Igrid_g.max()), 200)
Vlc_cmp = np.interp(Icmp, Igrid_lc, Vlc_mean)
Vg_cmp  = np.interp(Icmp, Igrid_g, Vg_mean)
dV = Vlc_cmp - Vg_cmp                                       # Eq. 23
with np.errstate(divide="ignore", invalid="ignore"):
    epsV = np.where(Vg_cmp > 0, dV / Vg_cmp, np.nan)        # Eq. 24

print("\nLow-cost vs potentiostat at fixed currents:")
print("  I(mA)   V_lowcost   V_gamry   dV(mV)   eps_V")
for ic in CMP_CURRENTS:
    vlc = np.interp(ic, Igrid_lc, Vlc_mean)
    vg  = np.interp(ic, Igrid_g, Vg_mean)
    print(f"  {ic*1e3:5.0f}    {vlc:7.3f}   {vg:7.3f}   {(vlc-vg)*1e3:6.1f}   {(vlc-vg)/vg*100:5.1f}%")

Pov_mean = np.mean([np.interp(OVERLAP_IMAX, d["Ib"], d["Vb"])*OVERLAP_IMAX for d in LC.values()])
epsP = (Pov_mean - GAMRY_MPP[0]) / GAMRY_MPP[0]
print(f"\nLow-cost power at {OVERLAP_IMAX*1e3:.0f} mA (mean) = {Pov_mean*1e3:.0f} mW   "
      f"Gamry mean MPP = {GAMRY_MPP[0]*1e3:.0f} mW  ->  eps_P = {epsP*100:+.0f}%")


# ===================== CELL 7 - repeatability statistics =====================
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
    print(f"{label:9s} = {v} +/- {u} {unit:3s}  (CV = {cv:.1f} %)")
    return {"quantity": label, "mean": m, "sd": s, "CV (%)": cv, "unit": unit}

print("\nRun-to-run repeatability (n = %d):" % len(runs))
rep = [
    repeat_stats(lc_summary["OCV (V)"],     "V_OC",    "V"),
    repeat_stats([d["V100"] for d in LC.values()], "V@100mA", "V"),
    repeat_stats(lc_summary["Papp (mW)"],   "P_app",   "mW"),
    repeat_stats(lc_summary["Iapp (mA)"],   "I_app",   "mA"),
]
rep_df = pd.DataFrame(rep)


# ===================== CELL 8 - Figure 5: time series ========================
REP_RUN = 6 if 6 in runs else runs[0]
d = LC[REP_RUN]
fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
ax2 = ax1.twinx()
lV = ax1.plot(d["t"], d["V"], color="#1f4e79", lw=1.4, label="voltage")[0]
lI = ax2.plot(d["t"], d["I"]*1e3, color="#b5651d", lw=1.2, label="current")[0]
t0, t1 = d["t"][d["i0"]], d["t"][d["i1"]]
tramp = d["t"][d["i0"]:d["i1"]+1][np.argmax(d["Id"] > 30e-3)]
ax1.axvspan(0, t0, color="0.90", zorder=0)
ax1.axvspan(t1, d["t"].max(), color="0.90", zorder=0)
ax1.axvspan(tramp, t1, color="#f3e0cf", alpha=0.5, zorder=0)
ax1.text((0+t0)/2, 1.07, "OCV\nhold", ha="center", va="center", fontsize=8, color="0.35")
ax1.text((t0+tramp)/2, 1.07, "stepped loads\n150-52 $\\Omega$", ha="center", va="center", fontsize=8, color="0.35")
ax1.text((tramp+t1)/2, 1.07, "ramp to 2 $\\Omega$\n(time-dependent)", ha="center", va="center", fontsize=8, color="#7a3d00")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Cell voltage (V)", color="#1f4e79")
ax2.set_ylabel("Discharge current (mA)", color="#b5651d")
ax1.tick_params(axis="y", colors="#1f4e79"); ax2.tick_params(axis="y", colors="#b5651d")
ax1.set_xlim(0, d["t"].max()); ax1.set_ylim(0.2, 1.15); ax2.set_ylim(0, None)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.legend(handles=[lV, lI], frameon=False, loc="center left")
fig.savefig(f"{OUTDIR}/figE_lowcost_timeseries.pdf"); fig.savefig(f"{OUTDIR}/figE_lowcost_timeseries.png")
plt.show()


# ===================== CELL 9 - Figure 6: polarisation overlay ===============
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for n in runs:
    ax.plot(LC[n]["Ib"]*1e3, LC[n]["Vb"], color=RUN_COLORS[n], alpha=0.9, label=f"Run {n}")
ax.fill_between(Igrid_g*1e3, Vg_mean-Vg_std, Vg_mean+Vg_std,
                color="0.55", alpha=0.35, zorder=0, label=r"potentiostat mean $\pm$1$\sigma$")
ax.plot(Igrid_g*1e3, Vg_mean, "k--", lw=2, label="potentiostat mean")
ax.set_xlabel("Discharge current (mA)"); ax.set_ylabel("Cell voltage (V)")
ax.set_xlim(0, None); ax.set_ylim(0.55, None)
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=3, frameon=False, loc="upper right")
fig.savefig(f"{OUTDIR}/figF_lowcost_polarisation.pdf"); fig.savefig(f"{OUTDIR}/figF_lowcost_polarisation.png")
plt.show()


# ===================== CELL 10 - Figure 7: power overlay =====================
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for n in runs:
    ax.plot(LC[n]["Ib"]*1e3, LC[n]["Pb"]*1e3, color=RUN_COLORS[n], alpha=0.9, label=f"Run {n}")
    k = int(np.argmax(LC[n]["Pb"]))
    ax.plot(LC[n]["Ib"][k]*1e3, LC[n]["Pb"][k]*1e3, "o", color=RUN_COLORS[n], ms=5)
ax.plot(Igrid_g*1e3, Pg_mean*1e3, "k--", lw=2, label="potentiostat mean")
ax.plot(GAMRY_MPP[2]*1e3, GAMRY_MPP[0]*1e3, "k*", ms=13, label="potentiostat MPP")
ax.set_xlabel("Discharge current (mA)"); ax.set_ylabel("Output power (mW)")
ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=3, frameon=False, loc="upper left")
fig.savefig(f"{OUTDIR}/figG_lowcost_power.pdf"); fig.savefig(f"{OUTDIR}/figG_lowcost_power.png")
plt.show()


# ============= CELL 11 - Figure 8: overlap comparison + residual  =============
fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
axA.fill_between(Igrid_g*1e3, Vg_mean-Vg_std, Vg_mean+Vg_std, color="0.55", alpha=0.30, zorder=0)
axA.plot(Igrid_g*1e3, Vg_mean, "k--", lw=2, label="potentiostat mean")
axA.fill_between(Igrid_lc*1e3, Vlc_mean-Vlc_std, Vlc_mean+Vlc_std, color="#1f4e79", alpha=0.20, zorder=0)
axA.plot(Igrid_lc*1e3, Vlc_mean, color="#1f4e79", lw=2, label="low-cost mean")
axA.set_ylabel("Cell voltage (V)")
axA.set_xlim(0, OVERLAP_IMAX*1e3); axA.set_ylim(0.6, None)
axA.legend(frameon=False, loc="upper right")
axB.axhline(0, color="0.6", lw=0.8)
axB.plot(Icmp*1e3, dV*1e3, color="#b5651d", lw=1.8)
axB.set_xlabel("Discharge current (mA)"); axB.set_ylabel(r"$\Delta V$ (mV)")
axB.xaxis.set_minor_locator(AutoMinorLocator())
fig.savefig(f"{OUTDIR}/figH_comparison_residual.pdf"); fig.savefig(f"{OUTDIR}/figH_comparison_residual.png")
plt.show()


# ============= CELL 12 - gas-production summary ==========
gas_volumes = {           # run : (H2 mL, O2 mL)
    1: (15.0, 7.5), 2: (15.0, 7.5), 3: (15.0, 7.5), 4: (15.0, 7.6),
    5: (15.1, 7.6), 6: (15.0, 7.5), 7: (15.0, 7.5), 8: (15.0, 7.5),
}
g_runs = [n for n in runs if n in gas_volumes]
gas = pd.DataFrame({"run": g_runs,
                    "H2 (mL)": [gas_volumes[n][0] for n in g_runs],
                    "O2 (mL)": [gas_volumes[n][1] for n in g_runs]})
TK = T_AMBIENT_C + 273.15
gas["H2:O2"]       = gas["H2 (mL)"] / gas["O2 (mL)"]
gas["n_H2 (mmol)"] = P_AMBIENT * gas["H2 (mL)"] * 1e-6 / (R_GAS * TK) * 1e3
gas["n_O2 (mmol)"] = P_AMBIENT * gas["O2 (mL)"] * 1e-6 / (R_GAS * TK) * 1e3
print(gas.round(3).to_string(index=False))
repeat_stats(gas["H2:O2"], "H2:O2", "")


# ============= CELL 13 ============================
lc_summary.round(3).to_csv(f"{OUTDIR}/table_lowcost_iv.csv", index=False)
rep_df.round(4).to_csv(f"{OUTDIR}/table_lowcost_repeatability.csv", index=False)
gas.round(4).to_csv(f"{OUTDIR}/table_gas.csv", index=False)
pd.DataFrame({"I (mA)": Icmp*1e3, "V_lowcost (V)": Vlc_cmp, "V_gamry (V)": Vg_cmp,
              "dV (mV)": dV*1e3, "eps_V (%)": epsV*100}
            ).round(4).to_csv(f"{OUTDIR}/table_comparison.csv", index=False)

try:
    import shutil
    from google.colab import files
    shutil.make_archive("lowcost_results", "zip", OUTDIR)
    files.download("lowcost_results.zip")
except Exception as e:
    print("Saved to", OUTDIR, "(download step skipped outside Colab):", e)
