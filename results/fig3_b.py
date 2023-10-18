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


tick_label = ['3', '4', '5']
#museum pre resiliency score
V2DN = [64.98 ,62.06, 58.192]
CE_PG = [280.28,231.18,224.57]
MADDPG = [142.977,121.27,116.71]
VDN = [140.53,131.41,125.93]
DQN = [158.95, 149.211, 140.86]
DRL = [219.45,202.68,181.84]
PD_Fac = [149.14,136.47,128.57]
MASAC = [196.9,189.23,177.43]
#office pre resilency score
V2DN = [141.11,130.92,122.95]
CE_PG = [317.35,310.75,301.65]
MADDPG = [281.82,214.08,202.21]
VDN = [278.57,199.89,198.63]
DQN = [263.45,282.71,231.05]
DRL = [311.4,282.11,234.74]
PD_Fac = [255.14,237.75,226.53]
MASAC = [285.93,278.5,274.84]

#museum during resiliency score
V2DN = [121.75,142.36,190]
CE_PG = [286.76,308.57,426.7]
MADDPG = [258.12,248.79,325]
VDN = [201.23,251.64,312.3]
DQN = [298.23,318.79,345.8]
DRL = [238.23,281.57,390]
PD_Fac = [233.53,208.96,282.8]
MASAC = [275.65,308.86,403.2]

#office during resilency score
V2DN = [198.08,244.63,332.8]
CE_PG = [370.83,476.42,518.17]
MADDPG = [358.56,303.47,467.98]
VDN = [343.58,324.21,405.11]
DQN = [483.83,372,521.67]
DRL = [370.86,426.32,650.42]
PD_Fac = [385,326.32,433.33]
MASAC = [381,454.58,569.08]

# V2DN = [37.7,33.93,29]
# CE_PG = [65.75,57.2,52.67]
# MADDPG = [61.05, 48.83,42.5]
# VDN = [51.21, 49.23 ,41.23]
# DQN = [67.7,58.63,44.58]
# DRL = [57.5,53.42,49]
bar_width = 0.1
bar_widthg = 0.105
tick_label = ['2', '3', '4']

plt.bar(np.arange(3), V2DN, bar_width, align="center", label="V2DN", alpha=1,ec='k', ls='-', lw=3,color=(163/255, 6/255, 67/255))
plt.bar(np.arange(3)+bar_widthg, CE_PG, bar_width, align="center", label="CE-PG", alpha=1,ec='k', ls='-', lw=3,color=(235/255, 96/255, 70/255))
plt.bar(np.arange(3)+2*bar_widthg, MADDPG, bar_width, align="center", label="MADDPG", alpha=1,ec='k', ls='-', lw=3,color=(98/255, 190/255, 166/255))
plt.bar(np.arange(3)+3*bar_widthg, VDN, bar_width, align="center", label="VDN", alpha=1,ec='k', ls='-', lw=3,color=(254/255, 251/255, 185/255))
plt.bar(np.arange(3)+4*bar_widthg, DQN, bar_width, align="center", label="DQN", alpha=1,ec='k', ls='-', lw=3,color=(205/255, 234/255, 157/255))
plt.bar(np.arange(3)+5*bar_widthg, DRL, bar_width, align="center", label="DRL", alpha=1,ec='k', ls='-', lw=3,color=(253/255, 186/255, 107/255))
plt.bar(np.arange(3)+6*bar_widthg, PD_Fac, bar_width, align="center", label="PD-FAC", alpha=1,ec='k', ls='-', lw=3,color=(236/255, 217/255, 207/255))
plt.bar(np.arange(3)+7*bar_widthg, MASAC, bar_width, align="center", label="MASAC", alpha=1,ec='k', ls='-', lw=3,color=(78/255, 146/255, 128/255))

plt.ylim(150,800)
plt.xlabel("Number of Robots ($N$)", weight='bold',fontsize=40)
plt.ylabel("Resiliency Score", weight='bold',fontsize=40)
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
plt.savefig('Dur OFFICE.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.show()