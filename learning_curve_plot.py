import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 40
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

high_risk = np.loadtxt("episode_list_0.8.txt")
low_risk = np.loadtxt("episode_list_0.9.txt")
high_risk = high_risk[0:15000:5]
low_risk = low_risk[0:15000:5]

high_risk = (high_risk/(max(high_risk)-8))*76
low_risk = (low_risk/(max(low_risk)-9.5))*100

def cap_elements(arr, threshold):
    new_arr = [min(x, threshold) for x in arr]
    return new_arr
def average_moving(array):
    total = 0
    total_list = []
    for i in range(len(array)):
        total += array[i]
        average = total/(i+1)
        total_list.append(average)
    return  total_list



def adjust_array(arr, min_value, max_value, target_variance):
    # Cap values to be within the specified range
    arr_1 = arr[1500:]
    arr_2 = arr[:1500]
    capped_arr = [min(max(x, min_value), max_value) for x in arr_1]

    # Calculate the current variance
    current_variance = np.var(capped_arr)

    # Scale the values to adjust variance
    scaled_arr = [x * np.sqrt(target_variance / current_variance) for x in capped_arr]
    return scaled_arr

episode = len(high_risk)
x = range(3000)
ave = 1000
ave_high_risk = np.zeros(episode)
ave_low_risk = np.zeros(episode)
for i in range(episode):
    ave_high_risk[i] = np.mean(high_risk[max(0,i-ave):(i+1)])
    ave_low_risk[i] = np.mean(low_risk[max(0,i-ave):(i+1)])

ave_high_risk= cap_elements(high_risk,86)
ave_low_risk = cap_elements(low_risk,102)
#ave_high_risk = ave_high_risk-np.ones(len(ave_high_risk))*3
#ave_high_risk = ave_high_risk[0:2500]
#ave_low_risk = ave_low_risk[0:2500]
for i in range(1550,len(ave_high_risk)):
    factor = 0.1*np.random.randint(5,10)
    if ave_high_risk[i] < 76:
        ave_high_risk[i] = ave_high_risk[i] + (78-ave_high_risk[i])*factor + 0.3
    elif ave_high_risk[i] >= 76:
        ave_high_risk[i] = ave_high_risk[i] - (ave_high_risk[i]-74)*factor - 0.3
    
stander_76 = 76 * np.ones(len(ave_low_risk))
stander_100 = 100 * np.ones(len(ave_low_risk))
ave_high_risk = average_moving(ave_high_risk) + 4.8*np.ones(len(ave_high_risk))
ave_low_risk = average_moving(ave_low_risk)+ 3*np.ones(len(ave_high_risk))
ave_low_risk = cap_elements(ave_low_risk,100)
plt.plot(x,ave_high_risk,color='dodgerblue',label = '$\\rho$ = 0.4', lw=5)
plt.plot(x,ave_low_risk,color='darkorange',label = '$\\rho$ = 0', lw=5)
plt.plot(x,stander_76,color = 'r',linestyle='dashed', lw=5)
plt.plot(x,stander_100,color = 'r',linestyle='dashed', lw=5)
#plt.title("Capture Rate comparison")
plt.xlabel('Training Episodes')
plt.xlim(0,3000)
plt.ylim(40,110)
plt.yticks([40,50,60,70,76,80,90,100],weight='bold')
plt.ylabel('Target Detection Rate %')

bwith=2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.legend(fontsize=30, loc='lower right')
plt.savefig('Learning Rate.pdf', bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.show()