import matplotlib.pyplot as plt
import numpy as np

nblks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
log2_nblks = np.log2(nblks)

execution_time = [997867, 1003362, 1006336, 1009783, 1014654, 1014553, 1972366, 2895758, 4837019, 9609594, 18176286, 36346661, 72592071, 145148659]
performance = [0.043041, 0.085612, 0.170717, 0.340269, 0.677270, 1.354675, 1.393646, 1.898487, 2.273118, 2.288362, 2.419662, 2.420055, 2.423431, 2.424023]
average_power = [87.344889, 87.236300, 93.575556, 101.673333, 117.564556, 148.589444, 155.249278, 178.288857, 201.682348, 206.060213, 215.193333, 218.598868, 220.031945, 218.800735]
power_efficiency = [0.000493, 0.000981, 0.001824, 0.003347, 0.005761, 0.009117, 0.008977, 0.010648, 0.011271, 0.011105, 0.011244, 0.011071, 0.011014, 0.011079]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('log2(nblks)')
ax1.set_ylabel('Execution Time (us)', color=color)
ax1.plot(log2_nblks, execution_time, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Performance (TFLOPS)', color=color)
ax2.plot(log2_nblks, performance, color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
color = 'tab:green'
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Average Power (W)', color=color)
ax3.plot(log2_nblks, average_power, color=color)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax1.twinx()
color = 'tab:orange'
ax4.spines['right'].set_position(('outward', 120))
ax4.set_ylabel('Power Efficiency (TFLOPS/W)', color=color)
ax4.plot(log2_nblks, power_efficiency, color=color)
ax4.tick_params(axis='y', labelcolor=color)

# Add a power function line (2^x)
ax5 = ax1.twinx()
color = 'tab:gray'
power_y = 2**(log2_nblks-2)
ax5.spines['right'].set_position(('outward', 180))
ax5.set_ylabel('Threads', color=color)
ax5.plot(log2_nblks, power_y, color=color)
ax5.tick_params(axis='y', labelcolor=color)


fig.tight_layout()


plt.show()
plt.savefig('result.png')
