# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:19:23 2025

@author: gangu
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np

path1= "aistats_VI_cRCMDP_CRS_ckl_0.01_v2_baseline_reduced.xlsx"
path2= "VF_CF_kl_lambda_RS_CRPO_aistats_rvi_baseline_comp_CKL_0.01_reduced_baseline.xlsx"
path3= "VF_CF_kl_lambda_RS_C_KL_0.01_RNPG_aistats_rvi_baseline_comp_reduced_baseline.xlsx"
env_ch = 0
baseline=4.5

env_nm = ["CRS","Garnet"]

rvi_dat = pd.read_excel(path1)
rnpg_dat = pd.read_excel(path2)
crpo_dat = pd.read_excel(path3)

vf,cf = rvi_dat['vf'],rvi_dat['cf']
rpng_vf,rpng_cf = rnpg_dat['vf'],rnpg_dat['cf']
crpo_vf,crpo_cf = crpo_dat['vf'],crpo_dat['cf']

plt.figure()
plt.plot(vf)
plt.plot(np.cumsum(vf)/np.arange(1,len(vf)+1))
plt.plot(rpng_vf)
plt.plot(crpo_vf)
plt.xlabel('Iterations')
plt.ylabel('Robust value function')
plt.legend(['RCVI (our)','Average RCVI (our)','RNPG','CRPO'])
plt.title('Robust Value function comparison')
plt.savefig('Comparison_plots_rho_1e-1_baseline_reduced_zoom'+env_nm[env_ch]+'vf.pdf')
plt.show()

x = np.arange(1,len(vf)+1)
fig, ax = plt.subplots(figsize=(7, 4))

# ----- Main plot -----
ax.plot(cf)
ax.plot(np.cumsum(cf) / np.arange(1, len(cf) + 1))
ax.plot(rpng_cf)
ax.plot(crpo_cf)
ax.plot(np.ones(len(cf)) * baseline, linewidth=3, linestyle=':')

ax.fill_between(x, 0, baseline, color='blue', alpha=0.1, label='Safe region')
ax.fill_between(x, baseline, 100, color='red', alpha=0.1, label='Unsafe region')

ax.set_xlabel('Iterations')
ax.set_ylabel('Robust value function')
ax.set_title('Robust cost function comparison')

ax.legend([
    'RCVI (our)',
    'Average RCVI (our)',
    'RNPG',
    'CRPO',
    'baseline',
    'Safe zone',
    'Unsafe zone'
],loc="upper left")

# ----- Zoomed inset -----
axins = inset_axes(
    ax,
    width="38%",
    height="38%",
    loc="upper right",
    borderpad=1.2
)

# Same curves inside inset
axins.plot(cf)
axins.plot(np.cumsum(cf) / np.arange(1, len(cf) + 1))
axins.plot(rpng_cf)
axins.plot(crpo_cf)
axins.plot(np.ones(len(cf)) * baseline, linewidth=2, linestyle=':')

# 🔍 Explicit zoom region
x1, x2 = 100, 850
y1, y2 = 4.3, 4.57

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Clean inset
axins.tick_params(labelsize=7)
axins.grid(True, alpha=0.3)

# ----- Callout: show zoomed region on main plot -----
ax.indicate_inset_zoom(
    axins,
    edgecolor="black",
    linewidth=1.2
)

ax.annotate(
    '',                              # no text, just arrow
    xy=(300, 4.57),                  # point INSIDE zoomed region (data coords)
    xycoords='data',
    xytext=(0.7, 0.8),             # point near inset (axes fraction)
    textcoords='axes fraction',
    arrowprops=dict(
        arrowstyle='->',
        linewidth=1.2,
        color='black'
    )
)

plt.tight_layout()
plt.savefig(
    'Comparison_plots_rho_1e-1_baseline_reduced_zoom' + env_nm[env_ch] + 'CF.pdf',
    bbox_inches='tight'
)
plt.show()