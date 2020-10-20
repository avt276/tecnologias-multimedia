import numpy as np 
from scipy import stats 
latencies = np.loadtxt("final_latencies.dat") # Name of latencies data file.
average_latency = np.average(latencies) 
print("average latency =", average_latency) 
max_latency = np.max(latencies) 
min_latency = np.min(latencies) 
maximum_absolute_deviation = max(max_latency - average_latency, average_latency - min_latency) 
print("maximum absolute deviation (jitter)=", maximum_absolute_deviation) 
correlation_coefficient = stats.pearsonr(latencies, np.roll(latencies, 1))[0] 
print("Pearson correlation coefficient =", correlation_coefficient) 
if correlation_coefficient < 0: 
  print("Correlation coefficient < 0: use 0 (no correlation between RTT samples) in your experiments") 
histogram = np.histogram(latencies) 
np.savetxt("final_histogram.dat", histogram[0]) # Name of histogram data file.