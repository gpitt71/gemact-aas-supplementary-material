# Remark, the execution of this script takes several hours
# due to the computing time measurement which runs several experiments.

from gemact.lossaggregation import Margins, Copula
from gemact.calculators import MCCalculator, AEPCalculator
import numpy as np
import pandas as pd
import timeit as timeit

#### script parameters
# local path to save
localpath = ''
explort_flag = True
dimensions = [2, 3, 4, 5]

# number of AEP iterations and MC simulations.
n_iter = [
    [7, 10, 13, 16],
    [7, 9, 11, 13],
    [4, 5, 6, 7],
    [3, 4, 5, 6]
    ]
n_simul = [10**4, 10**5, 10**6, 10**7]

# timeit parameters. Number of repetitions for computation time.
repetitions = 7 # numer of times the experiment is repeated
number = 100 # number of times the function is executed in a experiment.
# the raw results are expressed in centisecond, i.e. 10**(-2) second.

# values where the cdf is evaluated
sValues = [
    np.array([10.0**x for x in (0, 2, 4, 6)]),
    np.array([10.0**x for x in (0, 2, 4, 6)]),
    np.array([10.0**x for x in (1, 2, 3, 4)]),
    np.array([10.0**x for x in (1, 2, 3, 4)])
]

# containers to store computation accuracy outputs
acc_table = pd.DataFrame()
outputtxt = []


# create Margins and Copula objects
# Example from Arbenz, P., Embrechts, P., & Puccetti, G. (2011).
# The aep algorithm for the fast computation of the distribution of the sum of dependent random variables.
# Bernoulli, 17 (2), 562–591
margins = [
    Margins(
        dist=['genpareto']*2,
        par=[
            {'loc': 0, 'scale': 1/.9, 'c': 1/.9},
            {'loc': 0, 'scale': 1/1.8, 'c': 1/1.8}
            ]),
    Margins(
        dist=['genpareto']*3,
        par=[
            {'loc': 0, 'scale': 1/.9, 'c': 1/.9},
            {'loc': 0, 'scale': 1/1.8, 'c': 1/1.8},
            {'loc': 0, 'scale': 1/2.6, 'c': 1/2.6}
            ]),
    Margins(
        dist=['genpareto']*4,
        par=[
            {'loc': 0, 'scale': 1/.9, 'c': 1/.9},
            {'loc': 0, 'scale': 1/1.8, 'c': 1/1.8},
            {'loc': 0, 'scale': 1/2.6, 'c': 1/2.6},
            {'loc': 0, 'scale': 1/3.3, 'c': 1/3.3}
            ]),
    Margins(
        dist=['genpareto']*5,
        par=[
            {'loc': 0, 'scale': 1/.9, 'c': 1/.9},
            {'loc': 0, 'scale': 1/1.8, 'c': 1/1.8},
            {'loc': 0, 'scale': 1/2.6, 'c': 1/2.6},
            {'loc': 0, 'scale': 1/3.3, 'c': 1/3.3},
            {'loc': 0, 'scale': 1/4, 'c': 1/4}
            ])
    ]

copulas = [
    Copula(dist='clayton', par={'par': 1.2, 'dim': 2}),
    Copula(dist='clayton', par={'par': 0.4, 'dim': 3}),
    Copula(dist='clayton', par={'par': 0.2, 'dim': 4}),
    Copula(dist='clayton', par={'par': 0.3, 'dim': 5})
]

# Monte Carlo cdf function: wrapper for error and computation time
def MCCalculatorCdf(s, size, random_state, copula, margins):
    simul = MCCalculator.rvs(
        size=size,
        random_state=random_state,
        copula=copula,
        margins=margins
        ).reshape(-1, 1)
    simul = (simul <= s)
    prob = np.mean(simul, axis=0)
    return (prob, simul)

# reference value: taken from Arbenz's paper
rifVals = [
    # d0 = 2
    [0.315835041363441, 0.983690398913354, 0.999748719229367, 0.999996018908404],
    # d0 = 3
    [0.190859309689430, 0.983659549676444, 0.999748708770280, 0.999996018515584],
    # d0 = 4
    [0.833447516734442, 0.983412214152579, 0.997950264030106, 0.999742266243751],
    # d0 = 5
    [0.824132635126808, 0.983253494805448, 0.997930730055234, 0.999739803851201]
]
    
for i in range(len(dimensions)):
    dim = dimensions[i]
    print('dim = ' + str(dim))
    acc_table['ref_val_'+str(dim)] = rifVals[i]
    # AEP
    print('AEP')
    for ni in n_iter[i]:
        print('n_iter = ' + str(ni))
        # 1) computation accuracy
        p_aep = AEPCalculator.cdf(sValues[i], n_iter=ni, copula=copulas[i], margins=margins[i])
        err_aep = (rifVals[i] - p_aep) / rifVals[i]
        acc_table['val_aep_'+str(ni)+'_'+str(dim)] = p_aep
        acc_table['err_aep_'+str(ni)+'_'+str(dim)] = err_aep
        "{:.2e}".format(err_aep.astype(str))
        txt = ["{:.2e}".format(x) for x in p_aep]
        
        # 2) computation time
        stmt_aep = "AEPCalculator.core_cdf(" \
                                "x=sValues["+ str(i) + "][0],"\
                                "n_iter=" + str(ni) + ","\
                                "copula=copulas["+ str(i) + "],"\
                                "margins=margins["+ str(i) + "]," \
                                ")"
        out = timeit.repeat(stmt_aep,
                            repeat=repetitions,
                            number=number,
                            globals=locals())
        
        txt.insert(0, str(round(np.min(out) / number, 2)))
        txt.insert(0, '&AEP (' + str(ni) + ' iter.)')
        outputtxt.append('& '.join(txt) + '\\\\')

    # MC
    print('MC')
    random_state = int(10 + dim)
    for ns in n_simul:
        print('n_simul = ' + str(ns))
        # 1) computation accuracy
        p_mc, simul = MCCalculatorCdf(sValues[i], ns, random_state, copula=copulas[i], margins=margins[i])
        err_mc = (rifVals[i] - p_mc) / rifVals[i]
        emc_mc = np.std(simul, axis=0) / np.sqrt(ns)
        acc_table['val_mc_'+str(ns)+'_'+str(dim)] = p_mc
        acc_table['err_mc_'+str(ns)+'_'+str(dim)] = err_mc
        acc_table['emc_mc_'+str(ns)+'_'+str(dim)] = emc_mc
        random_state = (2 * random_state)
        # 2) computation time
        stmt_mc = "MCCalculatorCdf(" \
                    "s=sValues["+ str(i) + "][0],"\
                    "size=" + str(ns) + ","\
                    "random_state=random_state," \
                    "copula=copulas["+ str(i) + "],"\
                    "margins=margins["+ str(i) + "]," \
                    ")"
        out = timeit.repeat(stmt_mc,
                            repeat=repetitions,
                            number=number,
                            globals=locals())
        
        txt = ["{:.2e}".format(x) for x in err_mc]
        txt.insert(0, str(round(np.min(out) / number, 2)))
        txt.insert(0, '&MC (' + str(ns) + ' sim.)')
        outputtxt.append('& '.join(txt) + '\\\\')

# export results
if explort_flag:
    acc_table.to_csv(str(localpath + '\lossaggr_accuracy_vals.csv'), sep=",", index=False)
    pd.DataFrame(data = outputtxt, columns=['text']).to_csv(localpath + 'table_content.txt', index=False)
    print('finish')