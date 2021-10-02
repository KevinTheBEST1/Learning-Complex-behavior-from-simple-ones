import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('progress.log', sep=';')

plt.figure(figsize=(20,10))
plt.plot(data['average'])
#plt.plot(data['reward'])
plt.title('Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(['Average reward', 'Reward'], loc='upper right')
plt.show()
