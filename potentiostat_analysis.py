from google.colab import files
files.upload()

# ============================== CELL 1 ===============================
import os, re, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

T_AMBIENT_C = 21.2                 # ambient temperature during measurements (deg C)
P_AMBIENT   = 1.013e5              # ambient pressure used for gas conversion (Pa)
R_GAS       = 8.314462618          # ideal gas constant (J/mol/K)
OCV_WINDOW  = 30.0                 # averaging window at the end of the OCV hold (s)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10,
    "axes.linewidth": 0.9, "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})
RUN_COLORS = None                  # built in CELL 2 once the run list is known
OUTDIR = "figures_out"; os.makedirs(OUTDIR, exist_ok=True)


# ========================= CELL 2 ============================

def have_files():
    here = [f for f in os.listdir(".") if f.lower().endswith(".dta")]
    return any("_iv_" in f.lower() for f in here) and any("ocv" in f.lower() for f in here)

try:
    if not have_files():
        from google.colab import files
        print("Select all Gamry files (IV + OCV) ...")
        files.upload()
except Exception as e:
    print("Not running in Colab or upload skipped:", e)

iv_map, ocv_map = {}, {}
for f in os.listdir("."):
    if not f.lower().endswith(".dta"):
        continue
    m_iv  = re.search(r"_IV_(\d+)", f, re.IGNORECASE)
    m_ocv = re.search(r"ocv(\d+)",  f, re.IGNORECASE)
    if m_iv:
        iv_map[int(m_iv.group(1))] = f
    elif m_ocv:
        ocv_map[int(m_ocv.group(1))] = f

runs = sorted(set(iv_map) & set(ocv_map))        
print("Runs found:", runs)
missing_ocv = sorted(set(iv_map) - set(ocv_map))
missing_iv  = sorted(set(ocv_map) - set(iv_map))
if missing_ocv: print("WARNING: IV without matching OCV:", missing_ocv)
if missing_iv:  print("WARNING: OCV without matching IV:", missing_iv)
assert runs, "No paired IV/OCV runs were found - check the uploaded filenames."

RUN_COLORS = {n: c for n, c in zip(runs, plt.cm.viridis(np.linspace(0, 0.92, len(runs))))}


# ===================== CELL 3  =======================
def read_gamry_dta(path):
    """Read a Gamry .DTA file (latin-1, comma decimal separator).
    Returns a DataFrame with columns t (s), Vf (V), Im (A).
    Im is the raw Gamry current; it is negative while the cell discharges."""
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()

    # locate the data block: 'CURVE  TABLE  <N>' then a header row then a units row
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("CURVE") and "\tTABLE" in l)
    header = [h.strip() for h in lines[start + 1].split("\t")]
    col = {name: header.index(name) for name in ("T", "Vf", "Im")}

    t, vf, im = [], [], []
    for l in lines[start + 3:]:                 # skip header + units row
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


# ===================== CELL 4 — open-circuit voltage =========================
ocv_summary, ocv_curves = [], []
for n in runs:
    df  = read_gamry_dta(ocv_map[n])
    win = df[df["t"] >= df["t"].max() - OCV_WINDOW]
    ocv_summary.append({
        "run": n,
        "OCV (V)":        win["Vf"].mean(),
        "OCV sd (mV)":    win["Vf"].std(ddof=1) * 1e3,
        "hold time (s)":  df["t"].max(),
    })
    ocv_curves.append((n, df["t"].values, df["Vf"].values))
ocv_df = pd.DataFrame(ocv_summary)
print(ocv_df.round(4).to_string(index=False))


# ===================== CELL 5 — polarisation / power =========================
iv_summary, iv_curves = [], []
for n in runs:
    df = read_gamry_dta(iv_map[n])
    I  = -df["Im"].values            # positive discharge current (A)
    V  =  df["Vf"].values            # cell voltage (V)
    P  =  V * I                       # output power (W)
    k  = int(np.argmax(P))           # index of the maximum-power point
    iv_summary.append({
        "run": n,
        "V0 (V)":      V[0],          # voltage at scan start (~OCV)
        "Imax (mA)":   I.max() * 1e3,
        "Pmax (mW)":   P[k] * 1e3,
        "Vmpp (V)":    V[k],
        "Impp (mA)":   I[k] * 1e3,
    })
    iv_curves.append((n, I, V, P))
iv_df = pd.DataFrame(iv_summary)
print(iv_df.round(3).to_string(index=False))

# mean polarisation curve: interpolate every run onto a common current grid
Imax_common = min(I.max() for _, I, _, _ in iv_curves)  
Igrid = np.linspace(0.0, 0.99 * Imax_common, 400)
Vinterp = []
for _, I, V, _ in iv_curves:
    o = np.argsort(I)                          # ensure increasing current for interp
    Vinterp.append(np.interp(Igrid, I[o], V[o]))
Vinterp = np.array(Vinterp)
Vmean, Vstd = Vinterp.mean(0), Vinterp.std(0, ddof=1)
Pmean = Vmean * Igrid
kmean = int(np.argmax(Pmean))
print(f"\nMean curve  Pmax = {Pmean[kmean]*1e3:.1f} mW "
      f"at V = {Vmean[kmean]:.3f} V, I = {Igrid[kmean]*1e3:.0f} mA "
      f"(common range 0-{Imax_common*1e3:.0f} mA)")


# ===================== CELL 6 — repeatability statistics =====================
def round_up_unc(value, unc, sig=2):
    """Round the uncertainty UP to `sig` significant figures (math.ceil), and
    round the value to the same decimal place. Matches the thesis convention."""
    if unc == 0 or not np.isfinite(unc):
        return value, unc
    d = sig - 1 - int(math.floor(math.log10(abs(unc))))
    u = math.ceil(unc * 10**d) / 10**d
    v = round(value, d)
    return v, u

def repeat_stats(series, label, unit):
    x = np.asarray(series, float)
    m, s = x.mean(), x.std(ddof=1)
    cv = 100 * s / m
    v, u = round_up_unc(m, s)
    print(f"{label:8s} = {v} ± {u} {unit:3s}  (CV = {cv:.1f} %)")
    return {"quantity": label, "mean": m, "sd": s, "CV (%)": cv, "unit": unit}

print("Run-to-run repeatability (n = %d):" % len(iv_df))
rep = [
    repeat_stats(ocv_df["OCV (V)"],  "V_OC",  "V"),
    repeat_stats(iv_df["Pmax (mW)"], "P_max", "mW"),
    repeat_stats(iv_df["Vmpp (V)"],  "V_MPP", "V"),
    repeat_stats(iv_df["Impp (mA)"], "I_MPP", "mA"),
    repeat_stats(iv_df["Imax (mA)"], "I_max", "mA"),
]
rep_df = pd.DataFrame(rep)


# ===================== CELL 7 — Figure 1: OCV stabilisation ==================
fig, ax = plt.subplots(figsize=(7.0, 4.3))
for (n, t, v) in ocv_curves:
    ax.plot(t, v, color=RUN_COLORS[n], label=f"Run {n}")

tmax = max(t.max() for _, t, _ in ocv_curves)
ax.axvspan(tmax - OCV_WINDOW, tmax, color="0.85", zorder=0,
           label=f"final {OCV_WINDOW:.0f} s window")
ax.axhline(1.23, ls=":", color="0.4", lw=1)
ax.text(2, 1.232, "reversible voltage 1.23 V", fontsize=9, color="0.4", va="bottom")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Cell voltage (V)")
ax.set_xlim(0, tmax); ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.95))
fig.savefig(f"{OUTDIR}/figA_ocv_stabilisation.pdf")
fig.savefig(f"{OUTDIR}/figA_ocv_stabilisation.png")
plt.show()


# ===================== CELL 8 — Figure 2: polarisation curves ===============
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for (n, I, V, P) in iv_curves:
    ax.plot(I * 1e3, V, color=RUN_COLORS[n], alpha=0.85, label=f"Run {n}")
ax.fill_between(Igrid * 1e3, Vmean - Vstd, Vmean + Vstd,
                color="0.6", alpha=0.35, zorder=0, label=r"mean $\pm$ 1$\sigma$")
ax.plot(Igrid * 1e3, Vmean, "k--", lw=2, label="mean")
ax.set_xlabel("Discharge current (mA)"); ax.set_ylabel("Cell voltage (V)")
ax.set_xlim(0, None); ax.set_ylim(0.25, None)
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=3, frameon=False, loc="upper right")
fig.savefig(f"{OUTDIR}/figB_polarisation_curves.pdf")
fig.savefig(f"{OUTDIR}/figB_polarisation_curves.png")
plt.show()


# ===================== CELL 9 — Figure 3: power curves ======================
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for (n, I, V, P) in iv_curves:
    ax.plot(I * 1e3, P * 1e3, color=RUN_COLORS[n], alpha=0.85, label=f"Run {n}")
    k = int(np.argmax(P))
    ax.plot(I[k] * 1e3, P[k] * 1e3, "o", color=RUN_COLORS[n], ms=5)
ax.plot(Igrid * 1e3, Pmean * 1e3, "k--", lw=2, label="mean")
ax.plot(Igrid[kmean] * 1e3, Pmean[kmean] * 1e3, "k*", ms=13,
        label="mean MPP")
ax.set_xlabel("Discharge current (mA)"); ax.set_ylabel("Output power (mW)")
ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(ncol=3, frameon=False, loc="upper left")
fig.savefig(f"{OUTDIR}/figC_power_curves.pdf")
fig.savefig(f"{OUTDIR}/figC_power_curves.png")
plt.show()


# ============== CELL 10 — Figure 4: combined mean V-I and P-I ===============

fig, ax1 = plt.subplots(figsize=(7.0, 4.6))
ax2 = ax1.twinx()
l1 = ax1.plot(Igrid * 1e3, Vmean, color="#1f4e79", lw=2, label="voltage")[0]
ax1.fill_between(Igrid * 1e3, Vmean - Vstd, Vmean + Vstd,
                 color="#1f4e79", alpha=0.20)
l2 = ax2.plot(Igrid * 1e3, Pmean * 1e3, color="#b5651d", lw=2, label="power")[0]
l3 = ax2.plot(Igrid[kmean] * 1e3, Pmean[kmean] * 1e3, "*",
              color="#b5651d", ms=14, label="MPP")[0]
ax2.text(0.30, 0.40,
         f"$P_{{max}}$ = {Pmean[kmean]*1e3:.0f} mW\n"
         f"at {Vmean[kmean]:.2f} V, {Igrid[kmean]*1e3:.0f} mA",
         transform=ax2.transAxes, fontsize=10, color="#7a3d00",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
ax1.set_xlabel("Discharge current (mA)")
ax1.set_ylabel("Cell voltage (V)", color="#1f4e79")
ax2.set_ylabel("Output power (mW)", color="#b5651d")
ax1.tick_params(axis="y", colors="#1f4e79"); ax2.tick_params(axis="y", colors="#b5651d")
ax1.set_xlim(0, None); ax1.set_ylim(0.25, None); ax2.set_ylim(0, None)
ax1.legend(handles=[l1, l2, l3], frameon=False, loc="upper center")
fig.savefig(f"{OUTDIR}/figD_mean_polarisation_power.pdf")
fig.savefig(f"{OUTDIR}/figD_mean_polarisation_power.png")
plt.show()


# ============== CELL 11 — gas-production summary ==========

gas_volumes = {           
    1: (15.9, 7.6), 2: (15.1, 6.9), 3: (15.1, 7.5), 4: (15.2, 7.5),
    5: (15.5, 7.8), 6: (15.0, 7.4), 7: (15.2, 7.5),
    8: (15.0, 7.0),
}

g_runs = [n for n in runs if n in gas_volumes]
gas = pd.DataFrame({
    "run":     g_runs,
    "H2 (mL)": [gas_volumes[n][0] for n in g_runs],
    "O2 (mL)": [gas_volumes[n][1] for n in g_runs],
})
TK = T_AMBIENT_C + 273.15
gas["H2:O2"]      = gas["H2 (mL)"] / gas["O2 (mL)"]
gas["n_H2 (mmol)"] = P_AMBIENT * gas["H2 (mL)"] * 1e-6 / (R_GAS * TK) * 1e3
gas["n_O2 (mmol)"] = P_AMBIENT * gas["O2 (mL)"] * 1e-6 / (R_GAS * TK) * 1e3
print(gas.round(3).to_string(index=False))
repeat_stats(gas["H2:O2"], "H2:O2", "")


# ============== CELL 12 ===============
ocv_df.round(4).to_csv(f"{OUTDIR}/table_ocv.csv", index=False)
iv_df.round(3).to_csv(f"{OUTDIR}/table_iv_mpp.csv", index=False)
rep_df.round(4).to_csv(f"{OUTDIR}/table_repeatability.csv", index=False)
gas.round(4).to_csv(f"{OUTDIR}/table_gas.csv", index=False)

summary = (iv_df.merge(ocv_df[["run", "OCV (V)"]], on="run")
                 .loc[:, ["run", "OCV (V)", "Imax (mA)", "Vmpp (V)",
                          "Impp (mA)", "Pmax (mW)"]])
summary.round(3).to_csv(f"{OUTDIR}/table_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

try:
    import shutil
    from google.colab import files
    shutil.make_archive("potentiostat_results", "zip", OUTDIR)
    files.download("potentiostat_results.zip")
except Exception as e:
    print("Saved to", OUTDIR, "(download step skipped outside Colab):", e)
