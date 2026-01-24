# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:19:23 2025

@author: gangu
"""

import matplotlib .pyplot as plt
import pandas as pd
import numpy as np

path1= "aistats_VI_cRCMDP_RS_CKL_0.1.xlsx"
path2= "VF_CF_kl_lambda_Gar_C_KL_0.1_RNPG_aistats_rvi_baseline_comp.xlsx"
path3= "VF_CF_kl_lambda_Gar_CRPO_aistats_rvi_baseline_comp_CKL_0.01.xlsx"
env_ch = 0
baseline=4

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
#plt.plot(rpng_vf)
#plt.plot(crpo_vf)
plt.xlabel('Iterations')
plt.ylabel('Robust value function')
plt.legend(['RVI (our)','Average RVI (our)','RNPG'])
plt.title('Robust Value function comparison')
plt.savefig('Comparison_plots_rho_1e-1_'+env_nm[env_ch]+'vf.pdf')
plt.show()

x = np.arange(1,len(vf)+1)
plt.figure()
plt.plot(cf)
plt.plot(np.cumsum(cf)/np.arange(1,len(cf)+1))
#plt.plot(rpng_cf)
#plt.plot(crpo_cf)
plt.plot(np.ones(len(cf))*baseline,linewidth=3)
plt.fill_between(x, 0,baseline, color='blue', alpha=0.1, label='Safe region')
# Shade the region between y1 and y2 where y1 < y2
plt.fill_between(x,baseline, 100, color='red', alpha=0.1, label='UnSafe_region')
plt.xlabel('Iterations')
plt.ylabel('Robust value function')
plt.legend(['RVI (our)','Average RVI (our)','RNPG','baseline','Safe zone','Unsafe zone'])
plt.title('Robust cost function comparison')
plt.savefig('Comparison_plots_rho_1e-1_'+env_nm[env_ch]+'CF.pdf')
plt.show()