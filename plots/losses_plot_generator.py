"""
Scarica metriche di tipo 'losses' da run selezionate su Weights & Biases
e genera un'immagine 2x2 con le principali curve di training.
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

'''
#ROAD_CLUSTER
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
'''

#RANDOM
SELECTED_RUNS = [
    "mixed180-3UAV-powerlinear-randomEnv-100Fixgu",
    "mixed180-3UAV-power3-randomEnv-100Fixgu"
]

# 👇 Mappa per rinominare le run nella legenda
RUN_NAME_MAP = {
    "mixed180-3UAV-powerlinear-randomEnv-100Fixgu": "RandomEnv-powerlinear",
    "mixed180-3UAV-power3-randomEnv-100Fixgu": "RandomEnv-power3"
}

# 👇 Metriche di tipo "losses"
LOSS_METRICS = [
    "losses/value_loss",
    "losses/policy_loss",
    "losses/explained_variance",
    "losses/entropy",
]

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
    df["run_name"] = RUN_NAME_MAP.get(run.name, run.name)
    dataframes.append(df)

# ======================================
# 🎨 CREA FIGURA 2x2 (losses)
# ======================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

STEP_GRID = 100_000  # intervallo griglia step

for i, metric in enumerate(LOSS_METRICS):
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
    metric_label = metric.split("/")[-1]  # es. "policy_loss"
    ax.set_title(metric_label, fontsize=12)
    ax.set_xlabel("Step")
    ax.grid(True, which="both", axis="both", alpha=0.3)

    if plotted:
        x_min, x_max = ax.get_xlim()
        xticks = np.arange(0, x_max, STEP_GRID)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x/1000)}k" for x in xticks])
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="gray")

# Rimuove subplot vuoti se meno di 4 metriche
for j in range(len(LOSS_METRICS), len(axes)):
    fig.delaxes(axes[j])

# Titolo e salvataggio
#fig.suptitle(f"Andamento delle Losses – {PROJECT}", fontsize=16)
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("losses.png", dpi=200)
plt.show()