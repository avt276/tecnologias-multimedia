# Open files
with open('host-to-router/host-to-router_latencies.dat', 'r') as htrl, open('external-host-to-router/external-host-to-router_latencies.dat', 'r') as ehtrl, open('final_latencies.dat', 'w') as fl:
    # Loop both files
    for i, j in zip(htrl, ehtrl):
        #Sum values of each row.
        fl.write(str(float(i) + float(j)) + '\n')