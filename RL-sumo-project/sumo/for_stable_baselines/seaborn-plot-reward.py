import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


r1 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020120923-26-16reward.csv') - 2048
r2 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121000-22-38reward.csv') - 2048
r3 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121001-19-31reward.csv') - 2048
r4 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121002-16-14reward.csv') - 2048
# r5 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120820-12-39vmean.csv')
# r6 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-33-37vmean.csv')
# r7 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-48-30vmean.csv')
# r8 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-03-27vmean.csv')
# r9 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-18-28vmean.csv')
# r10 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-33-25vmean.csv')

r = [r1.to_numpy().T, r2.to_numpy().T, r3.to_numpy().T,
     r4.to_numpy().T
     ] #, r5.to_numpy().T, r6.to_numpy().T, r7.to_numpy().T, r8.to_numpy().T, r9.to_numpy().T, r10.to_numpy().T
r_episode = [[], [], [], []]  #, [], [], [], [], [], []

# for i in range(len(r)):
#     for j in range(300000 // 5000):
#         r_episode[i].append(r[i].to_numpy()[j*5000: (j+1)*5000].mean())

episode = range(60)
r_epi_stack = np.vstack((r[0], r[1], r[2], r[3],))  # r[4], r[5], r[6], r[7], r[8], r[9]
data = pd.DataFrame(r_epi_stack).melt(var_name='episode', value_name='episode mean reward')
# data = pd.concat((data1, data2), axis=1)

data[''] = 'velocity mean'    # add legend

plt.plot(np.linspace(0,199,200), -685*np.ones((200,)), color='black', linestyle='--')

sns.lineplot(x='episode', y='episode mean reward', hue='', style='', data=data)
plt.legend(['Avg. IDM platoon performance','CAV Policy'], loc='lower right')

plt.show()
