import numpy as np
from subprocess import call

fo = open("params.txt", "w")

expt = 1

Ni = 784
No = 10
snrs = np.logspace(-1,1,1)
alphas = np.linspace(.5,1.5,1)
depths = [0,1]
weightscales = np.logspace(-1,1,1)
rseeds = np.arange(5)

i = 1
for snr in snrs:
    for alpha in alphas:
        P = int(np.round(alpha*Ni))
        for d in depths:
            for ws in weightscales:
                for rseed in rseeds:
                    fo.write("-rseed %d -weightscale %g -snr %g -lr .01 -numsamples %d -numinputs %d -numoutputs %d -depth %d -epochs 5 -savefile /n/home13/asaxe/gendynamics/results/expt%d/res%d.npz\n" % (rseed, ws, snr, P, Ni, No, d, expt, i))
                    i = i+1
                    
fo.close()

call("python run_odyssey_array.py -expt %d -mem 1500 -partition serial_requeue -paramfile params.txt -mock" % expt, shell=True)
