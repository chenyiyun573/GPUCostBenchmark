import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# create data frames
idle_data = pd.DataFrame({
    'GPU0': [59.24538503217067, 77.25706980572933, 196.38653478751414], 
    'GPU1': [55.728829664747714, 74.0717033256503, 189.7335806694246], 
    'GPU2': [60.38429529292244, 78.80815146526176, 203.6236995110944], 
    'GPU3': [58.159711818489676, 76.37816924596642, 195.98452764197066], 
    'GPU4': [57.08861869285474, 74.32255054329931, 186.6923553967657], 
    'GPU5': [53.57722011513715, 71.32420777082648, 186.78030876269275], 
    'GPU6': [57.68229664747714, 75.68234408956206, 203.0080503948853], 
    'GPU7': [54.21986454453099, 72.42066381297332, 194.904450921399],
})

# create x-axis values
x = np.arange(3)

# create the plot
plt.figure(figsize=(10, 8))

for column in idle_data.columns:
    plt.plot(x, idle_data[column], marker='o', linewidth=2, label=column)

plt.title('GPUs Power under Different Stress Tests')
plt.xlabel('Stress Test Type')
plt.ylabel('Power')
plt.xticks(x, ['Idle', 'Small Stress', 'Big Stress'])
plt.legend(title='GPU')

# save the figure instead of displaying it
plt.savefig('path_to_save_figure.png')
