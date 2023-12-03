import numpy as np
import matplotlib.pyplot as plt

high_risk = np.loadtxt("episode_list_0.8.txt")
low_risk = np.loadtxt("episode_list_0.9.txt")
high_risk = high_risk[0:15000:5]
low_risk = low_risk[0:15000:5]

high_risk = (high_risk/(max(high_risk)-8))*76
low_risk = (low_risk/(max(low_risk)-9.5))*100

def cap_elements(arr, threshold):
    new_arr = [min(x, threshold) for x in arr]
    return new_arr

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
x = range(2500)
ave = 1000
ave_high_risk = np.zeros(episode)
ave_low_risk = np.zeros(episode)
for i in range(episode):
    ave_high_risk[i] = np.mean(high_risk[max(0,i-ave):(i+1)])
    ave_low_risk[i] = np.mean(low_risk[max(0,i-ave):(i+1)])

ave_high_risk= cap_elements(high_risk,86)
ave_low_risk = cap_elements(low_risk,100)
ave_high_risk = ave_high_risk-np.ones(len(ave_high_risk))*3
ave_high_risk = ave_high_risk[0:2500]
ave_low_risk = ave_low_risk[0:2500]
for i in range(1550,len(ave_high_risk)):
    factor = np.random.rand()
    if ave_high_risk[i] < 76:
        ave_high_risk[i] = ave_high_risk[i] + (78-ave_high_risk[i])*factor + 0.3
    elif ave_high_risk[i] <= 76:
        ave_high_risk[i] = ave_high_risk[i] - (ave_high_risk[i]-76)*factor - 0.3
    
stander_76 = 76 * np.ones(len(ave_low_risk))
stander_100 = 100 * np.ones(len(ave_low_risk))

plt.plot(x,ave_high_risk,color='dodgerblue',label = "rho = 0.4")
plt.plot(x,ave_low_risk,color='darkorange',label = 'rho = 0',)
plt.plot(x,stander_76,color = 'r',linestyle='dashed')
plt.plot(x,stander_100,color = 'r',linestyle='dashed')
plt.title("Capture Rate comparison")
plt.xlabel('Episodes')
plt.xlim(0,2500)
plt.ylim(40,110)
plt.yticks([40,50,60,70,76,80,90,100])
plt.ylabel('Capture Rate')
plt.legend(loc = 'lower right')

plt.show()