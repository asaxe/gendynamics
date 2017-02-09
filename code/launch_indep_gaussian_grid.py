import numpy as np
from subprocess import call

fo = open("params.txt", "w")

expt = 1

Ni = 1000
No = 10
snrs = np.logspace(np.log10(.05),np.log10(5),7)
alphas = np.linspace(.5,1.5,7)
depths = [0,1]
weightscales = np.linspace(.1,10,5)
rseeds = np.arange(10)

i = 1
for snr in snrs:
    for alpha in alphas:
        P = int(np.round(alpha*Ni))
        for d in depths:
            for ws in weightscales:
                for rseed in rseeds:
                    fo.write("-rseed %d -weightscale %g -snr %g -lr .01 -numsamples %d -numinputs %d -numoutputs %d -depth %d -epochs 5000 -savefile /n/home13/asaxe/gendynamics/results/expt%d/data/res%d.npz\n" % (rseed, ws, snr, P, Ni, No, d, expt, i))
                    i = i+1
                    
fo.close()

call("python run_odyssey_array.py -expt %d -mem 2000 -partition serial_requeue -paramfile params.txt" % expt, shell=True)
