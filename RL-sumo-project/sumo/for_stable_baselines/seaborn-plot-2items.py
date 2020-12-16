import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


r1 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020120923-26-16vmean.csv')
r2 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121000-22-38vmean.csv')
r3 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121001-19-31vmean.csv')
r4 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121002-16-14vmean.csv')
# r5 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120820-12-39vmean.csv')
# r6 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-33-37vmean.csv')
# r7 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-48-30vmean.csv')
# r8 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-03-27vmean.csv')
# r9 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-18-28vmean.csv')
# r10 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-33-25vmean.csv')

p1 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020120923-26-16vstd.csv')
p2 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121000-22-38vstd.csv')
p3 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121001-19-31vstd.csv')
p4 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/6-rl-cars/2020121002-16-14vstd.csv')
# p5 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120820-12-39vstd.csv')
# p6 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-33-37vstd.csv')
# p7 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120800-48-30vstd.csv')
# p8 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-03-27vstd.csv')
# p9 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-18-28vstd.csv')
# p10 = pd.read_csv('/home/sandymark/SnG_wave_TrainResult/2020120801-33-25vstd.csv')

r = [r1.to_numpy().T, r2.to_numpy().T, r3.to_numpy().T,
     r4.to_numpy().T] #, r5.to_numpy().T, r6.to_numpy().T, r7.to_numpy().T, r8.to_numpy().T, r9.to_numpy().T, r10.to_numpy().T
r_episode = [[], [], [], []]  #, [], [], [], [], [], []

p = [p1.to_numpy().T, p2.to_numpy().T, p3.to_numpy().T,
     p4.to_numpy().T]  #, p5.to_numpy().T, p6.to_numpy().T, p7.to_numpy().T, p8.to_numpy().T, p9.to_numpy().T, p10.to_numpy().T
p_episode = [[], [], [], []]  #, [], [], [], [], [], []

# for i in range(len(r)):
#     for j in range(300000 // 5000):
#         r_episode[i].append(r[i].to_numpy()[j*5000: (j+1)*5000].mean())

episode = range(60)
r_epi_stack = np.vstack((r[0], r[1], r[2], r[3]))  #, r[4], r[5], r[6], r[7], r[8], r[9]
p_epi_stack = np.vstack((p[0], p[1], p[2], p[3]))  #, p[4], p[5], p[6], p[7], p[8], p[9]
data1 = pd.DataFrame(r_epi_stack).melt(var_name='episode', value_name='average platoon speed (m/s) f')
data2 = pd.DataFrame(p_epi_stack).melt(var_name='episode', value_name='std. of platoon speed')
# data = pd.concat((data1, data2), axis=1)

data1[''] = 'velocity mean'    # add legend
data2[''] = 'velocity std.'


plt.plot(np.linspace(0,199,200), 2.78*np.ones((200,)), color='g', linestyle='--')
plt.plot(np.linspace(0,199,200), 1.95*np.ones((200,)), color='r', linestyle='--')

sns.lineplot(x='episode', y='average platoon speed (m/s) f', hue='', style='', data=data1, palette=sns.color_palette('Greens', 1))
sns.lineplot(x='episode', y='std. of platoon speed', hue='', style='', data=data2, palette=sns.color_palette('Reds', 1))
plt.legend(['Avg. IDM platoon v_mean','Avg. IDM platoon v_std.', 'Mixed Platoon v_mean', 'Mixed Platoon v_std.'], loc=(135/200, 0.8/2.8))
# sns.lineplot(x='episode', y='episode velocity STD', hue='', style='', data=data2, palette=sns.color_palette("Reds", 1))
# sns.lineplot(palette=)
plt.show()
