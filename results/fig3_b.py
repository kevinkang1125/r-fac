import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 40
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2


w_NC_mean = [50.884,38.445,34.45]
PatrolGRAPHstar_mean = [56.43,52.29,46.157]
PatrolGRAPHA_mean = [53.455,34.689,30.71]
SCTSP_mean = [53,35.987,36.37]
DQN_mean = [58.996,50.854,31.8]
EMP_mean = [33.755,	29.71,22.68]

w_NC_mean = [54.38,42.366,36]
PatrolGRAPHstar_mean = [82.056,56.3,45.44]
PatrolGRAPHA_mean = [46.57,36.617,31.34]
SCTSP_mean = [50.513,39.34,37.07]
DQN_mean = [63.89,54.456,47.4]
EMP_mean = [36.296,	26.551,24.174]
bar_width = 0.1
bar_widthg = 0.105
tick_label = ['2', '3', '4']

V2DN = [39,34.74,25.968]
CE_PG = [56.5,54.76,37.09]
MADDPG = [55.029,38.33,34.079]
VDN = [53.23,40.3,30.307]
DQN = [70.06,44.84,37.3]
DRL = [56.505,50,45.025]

# V2DN = [37.7,33.93,29]
# CE_PG = [65.75,57.2,52.67]
# MADDPG = [61.05, 48.83,42.5]
# VDN = [51.21, 49.23 ,41.23]
# DQN = [67.7,58.63,44.58]
# DRL = [57.5,53.42,49]
bar_width = 0.1
bar_widthg = 0.105
tick_label = ['3', '4', '5']

plt.bar(np.arange(3), V2DN, bar_width, align="center", label="V2DN", alpha=1,ec='k', ls='-', lw=3,color=(78/255, 171/255, 144/255))
plt.bar(np.arange(3)+bar_widthg, CE_PG, bar_width, align="center", label="CE-PG", alpha=1,ec='k', ls='-', lw=3,color=(142/255, 182/255, 156/255))
plt.bar(np.arange(3)+2*bar_widthg, MADDPG, bar_width, align="center", label="MADDPG", alpha=1,ec='k', ls='-', lw=3,color=(237/255, 221/255, 195/255))
plt.bar(np.arange(3)+3*bar_widthg, VDN, bar_width, align="center", label="VDN", alpha=1,ec='k', ls='-', lw=3,color=(238/255, 191/255, 109/255))
plt.bar(np.arange(3)+4*bar_widthg, DQN, bar_width, align="center", label="DQN", alpha=1,ec='k', ls='-', lw=3,color=(217/255, 79/255, 51/255))
plt.bar(np.arange(3)+5*bar_widthg, DRL, bar_width, align="center", label="DRL", alpha=1,ec='k', ls='-', lw=3,color=(131/255, 64/255, 38/255))

plt.ylim(20,120)
plt.xlabel("Number of Robots ($N$)", weight='bold',fontsize=40)
plt.ylabel("Capture Time", weight='bold',fontsize=40)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xticks(np.arange(3)+bar_widthg*2.5,tick_label)
plt.title("OFFICE During Execution rho = 0.2")

bwith=2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.legend(ncol=2, frameon=False, fontsize=30, loc='upper left')
plt.savefig('DUR OFFICE.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.show()