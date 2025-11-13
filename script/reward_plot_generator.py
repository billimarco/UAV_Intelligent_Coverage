"""
Scarica metriche specifiche da run selezionate di Weights & Biases
e ricrea i 9 grafici del pannello 'Charts', con limiti Y opzionali e nomi run personalizzati.
Requisiti:
    pip install wandb pandas matplotlib
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym_cruising_v2.utils.runtime_utils as utils

args = utils.parse_args()

# ======================================
# ⚙️ CONFIGURAZIONE
# ======================================
ENTITY = args.wandb_entity
PROJECT = args.wandb_project_name
GROUP = None

# 👇 Run da includere
'''
#UNIFORM
SELECTED_RUNS = [
    "mixed180-3UAV-powerlinear-uniform-100Fixgu",
    "mixed180-3UAV-powerlinear-uniform-Randomgu",
]

# 👇 Mappa per rinominare le run nella legenda
RUN_NAME_MAP = {
    "mixed180-3UAV-powerlinear-uniform-100Fixgu": "Uniform-NoRAU-powerlinear",
    "mixed180-3UAV-powerlinear-uniform-Randomgu": "Uniform-RAU-powerlinear",
}
'''

'''
#CLUSTERED
SELECTED_RUNS = [
    "mixed180-3UAV-powerlinear-1cluster-fleet-100Fixgu",
    "mixed180-3UAV-power3-1cluster-fleet-100Fixgu",
    "mixed180-3UAV-power3-1cluster-RandomMov-100Fixgu",
    "mixed180-3UAV-power3-[1,2]cluster-fleet-100Fixgu",
    "mixed180-3UAV-power3-[1,2]cluster-RandomMov-100Fixgu",
    "mixed180-3UAV-power3-2cluster-fleet-100Fixgu",
    "mixed180-3UAV-power3-2cluster-RandomMov-100Fixgu"
]

# 👇 Mappa per rinominare le run nella legenda
RUN_NAME_MAP = {
    "mixed180-3UAV-powerlinear-1cluster-fleet-100Fixgu": "1Cluster-Fleet-powerlinear",
    "mixed180-3UAV-power3-1cluster-fleet-100Fixgu": "1Cluster-Fleet-power3",
    "mixed180-3UAV-power3-1cluster-RandomMov-100Fixgu": "1Cluster-Random-power3",
    "mixed180-3UAV-power3-[1,2]cluster-fleet-100Fixgu": "1-2Cluster-Fleet-power3",
    "mixed180-3UAV-power3-[1,2]cluster-RandomMov-100Fixgu": "1-2Cluster-Random-power3",
    "mixed180-3UAV-power3-2cluster-fleet-100Fixgu": "2Cluster-Fleet-power3",
    "mixed180-3UAV-power3-2cluster-RandomMov-100Fixgu": "2Cluster-Random-power3",
}
'''

'''
#ROAD
SELECTED_RUNS = [
    "mixed180-3UAV-powerlinear-1road-poisson-100Fixgu",
    "mixed180-3UAV-power3-1road-poisson-100Fixgu",
    "mixed180-3UAV-powerlinear-1road-poisson-Randomgu",
    "mixed180-3UAV-power3-1road-poisson-Randomgu"
]

# 👇 Mappa per rinominare le run nella legenda
RUN_NAME_MAP = {
    "mixed180-3UAV-powerlinear-1road-poisson-100Fixgu" : "1Road-NoRAU-powerlinear",
    "mixed180-3UAV-power3-1road-poisson-100Fixgu" : "1Road-NoRAU-power3",
    "mixed180-3UAV-powerlinear-1road-poisson-Randomgu" : "1Road-RAU-powerlinear",
    "mixed180-3UAV-power3-1road-poisson-Randomgu" : "1Road-RAU-power3"
}
'''

SELECTED_RUNS = [
    "mixed180-3UAV-power3-1road-1cluster-poisson-100Fixgu",
    "mixed180-3UAV-powerlinear-1road-1cluster-poisson-100Fixgu",
    "mixed180-3UAV-power3-1road-[1,2]cluster-poisson-100Fixgu",
    "mixed180-3UAV-powerlinear-1road-[1,2]cluster-poisson-100Fixgu",
    "mixed180-3UAV-power3-1road-2cluster-poisson-100Fixgu",
    "mixed180-3UAV-powerlinear-1road-2cluster-poisson-100Fixgu"
]

# 👇 Mappa per rinominare le run nella legenda
RUN_NAME_MAP = {
    "mixed180-3UAV-power3-1road-1cluster-poisson-100Fixgu": "1Road-1cluster-power3",
    "mixed180-3UAV-powerlinear-1road-1cluster-poisson-100Fixgu": "1Road-1cluster-powerlinear",
    "mixed180-3UAV-power3-1road-[1,2]cluster-poisson-100Fixgu": "1Road-[1,2]cluster-power3",
    "mixed180-3UAV-powerlinear-1road-[1,2]cluster-poisson-100Fixgu": "1Road-[1,2]cluster-powerlinear",
    "mixed180-3UAV-power3-1road-2cluster-poisson-100Fixgu": "1Road-2cluster-power3",
    "mixed180-3UAV-powerlinear-1road-2cluster-poisson-100Fixgu": "1Road-2cluster-powerlinear"
}

# 👇 Metriche da plottare
METRICS = [
    "test/mean_rcr",
    "test/std_rcr",
    "test/mean_reward",
    "test/std_reward",
    "test/mean_out_area",
    "test/mean_collision",
    "test/mean_spatial_coverage",
    "test/mean_exploration_incentive",
    "test/mean_gu_coverage",
]

# ======================================
# 🎚️ LIMITI Y e SWITCH Booleano
# ======================================
YLIMS = {
    "test/mean_rcr": (0.0, 1.0),
    "test/std_rcr": (0.0, 1.0),
    "test/mean_reward": (-1500.0, 3000.0),
    "test/std_reward": (-1500.0, 3000.0),
    "test/mean_out_area": (0.0, 50.0),
    "test/mean_collision": (0.0, 50.0),
    "test/mean_spatial_coverage": (-1500.0, 0.0),
    "test/mean_exploration_incentive": (0.0, 1500.0),
    "test/mean_gu_coverage": (0.0, 1500.0),
}

# 👇 True = usa limiti Y manuali; False = scala automatica
USE_YLIMS = {
    "test/mean_rcr": False,
    "test/std_rcr": False,
    "test/mean_reward": False,
    "test/std_reward": False,
    "test/mean_out_area": False,
    "test/mean_collision": False,
    "test/mean_spatial_coverage": False,
    "test/mean_exploration_incentive": False,
    "test/mean_gu_coverage": False,
}

# ======================================
# 🔗 CONNESSIONE ALL'API
# ======================================
api = wandb.Api()
filters = {"group": GROUP} if GROUP else {}
runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)

selected_runs = [r for r in runs if r.name in SELECTED_RUNS or r.id in SELECTED_RUNS]

print(f"📦 Trovate {len(selected_runs)} run selezionate:")
for r in selected_runs:
    print("  -", r.name)

if not selected_runs:
    raise ValueError("❌ Nessuna run trovata! Controlla nomi o ID in SELECTED_RUNS.")

# ======================================
# 📊 SCARICA I DATI DI OGNI RUN
# ======================================
dataframes = []
for run in selected_runs:
    print(f"⬇️  Scarico dati per run: {run.name}")
    df = run.history(samples=10000)
    print(f"   → {len(df)} righe, colonne: {list(df.columns)}")
    df["run_name"] = RUN_NAME_MAP.get(run.name, run.name)  # usa nome rinominato
    dataframes.append(df)

# ======================================
# 🎨 CREA FIGURA 3x3
# ======================================
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

STEP_GRID = 100_000  # intervallo griglia in step

for i, metric in enumerate(METRICS):
    ax = axes[i]
    plotted = False

    for df in dataframes:
        # trova asse X
        x_col = "_step" if "_step" in df.columns else (
            "global_step" if "global_step" in df.columns else None)
        if x_col is None or metric not in df.columns:
            continue

        df_clean = df[[x_col, metric]].dropna()
        if df_clean.empty:
            continue

        ax.plot(df_clean[x_col], df_clean[metric],
                label=df["run_name"].iloc[0], linewidth=1.8)
        plotted = True

    # Titoli e assi
    metric_label = metric.split("/")[-1]  # 🔹 rimuove 'test/' o altri prefissi
    ax.set_title(metric_label, fontsize=12)
    ax.set_xlabel("Step")
    ax.grid(True, which="both", axis="both", alpha=0.3)

    if plotted:
        x_min, x_max = ax.get_xlim()
        xticks = np.arange(0, x_max, STEP_GRID)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x/1000)}k" for x in xticks])
        ax.legend(fontsize=8, loc="best")

        # 🔹 Applica limiti Y solo se USE_YLIMS[metric] è True
        if USE_YLIMS.get(metric, False) and metric in YLIMS:
            ymin, ymax = YLIMS[metric]
            ax.set_ylim(ymin, ymax)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="gray")

# Rimuove subplot vuoti
for j in range(len(METRICS), len(axes)):
    fig.delaxes(axes[j])

#fig.suptitle(f"Ambiente con distribuzione uniforme degli utenti", fontsize=16)
#plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("reward.png", dpi=200)
plt.show()
