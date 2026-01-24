# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:23:27 2025

@author: Sourav
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path1="aistats_VI_cRCMDP_Garnet.xlsx"
path2 = "VF_CF_kl_lambda_Gar_RNPG_aistats_rvi_baseline_comp.xlsx"
path3 = "VF_CF_kl_lambda_Gar_CRPO_aistats_rvi_baseline_comp.xlsx"
baseline = 15

data1 = pd.read_excel(path1)
data2 = pd.read_excel(path2)
data3 = pd.read_excel(path3)
vf,cf = data1['vf'],data1['cf']
rnvf,rncf = data2['vf'],data2['cf']
crvf,crcf = data3['vf'],data3['cf']

x = np.arange(1,len(vf)+1)

plt.figure()
plt.rcParams["font.weight"] = "bold"
plt.plot(vf,color='blue')
plt.plot(np.cumsum(vf)/np.arange(1,len(vf)+1),color='orange',linewidth=3)
plt.plot(rnvf,alpha=0.6,color='green')
#plt.plot(np.cumsum(rnvf)/np.arange(1,len(rnvf)+1),linewidth=3,color='brown')
plt.plot(crvf,alpha=0.6,color='red')
#plt.plot(np.cumsum(crvf)/np.arange(1,len(crvf)+1),linewidth=3,color='violet')
plt.xlabel('Iteration',fontweight='bold')
plt.ylabel('Robust value function',fontweight='bold')
plt.title('Value function for RVI in augmented space',fontweight='bold')
plt.legend(['Actual vf(our)','Averaged_vf(our)','RNPG','Robust Crpo'],prop={'weight':'bold'})
plt.savefig('Aistats_VI_vf_Garnet.pdf')
plt.show()

plt.figure()
plt.rcParams["font.weight"] = "bold"
plt.plot(cf,color='blue')
plt.plot(np.cumsum(cf)/np.arange(1,1001),linewidth = 3,color='orange')
plt.plot(rncf,color='green',alpha =0.6)
#plt.plot(np.cumsum(rncf)/np.arange(1,len(rncf)+1),linewidth=3,color='brown')
plt.plot(crcf,alpha=0.6,color='red')
#plt.plot(np.cumsum(crcf)/np.arange(1,len(crcf)+1),linewidth=3,color='violet')
plt.fill_between(x, 0,baseline, color='blue', alpha=0.1, label='Safe region')
plt.fill_between(x,baseline, 83, color='red', alpha=0.1, label='UnSafe_region')
plt.plot(np.ones(len(cf))*baseline,color='black',linewidth=3)
plt.xlabel('Iteration',fontweight='bold')
plt.ylabel('Robust cost function',fontweight='bold')
plt.title('Cost function for RVI in augmented space',fontweight='bold')
plt.legend(['Actual vf(our)','Averaged_vf(our)','RNPG','Robust Crpo','Safe region','Unsafe region','baseline'],prop={'weight':'bold'})
plt.savefig('Aistats_VI_cf_Garnet.pdf')
plt.show()



