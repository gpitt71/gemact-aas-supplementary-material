# Remark, the execution of this script takes several hours
# due to the computing time measurement which runs several experiments.

from gemact.lossmodel import Severity, Frequency, LossModel, PolicyStructure, Layer
from gemact.calculators import LossModelCalculator as Calculator
import numpy as np
import pandas as pd
import timeit as timeit
from aggregate import build, qd

# script parameters
# local path to save
localpath = 'C:/Users/luini/OneDrive/Desktop/lossmodel_study/'
explort_flag = True

# severity parameters
mu = 84541.68  # target mean
sigma = 177728.3  # target std

a = 1 + (sigma / mu) ** 2
shape = np.sqrt(np.log(a))
scale = mu / np.sqrt(a)

sev_par = {'scale': scale, 'shape':shape}
sev_dist_name = 'lognormal'
freq_par = {'mu': 3}
freq_dist_name = 'poisson'

############################################################################################################################
# Comparison of aggr. loss distribution methods, computing time and accuracy
############################################################################################################################

# Parametric assumptions taken from examples in
# Parodi, P. (2014). Pricing in general insurance (first ed.), pag. 262-266

# timeit parameters. Number of repetitions for computation time.
repetitions = 7 # numer of times the experiment is repeated
number = 100 # number of times the function is executed in a experiment.

# policystructure parmateres
deductible = 10000
cover = float('inf')

n_sev_nodes = [2**14, 2**16, 2**18]
n_sev_nodes_str = ['2**14', '2**16', '2**18']
n_aggr_nodes =  [2**14, 2**16, 2**18]
discr_step = [50, 100, 200, 400]
n_sim_mc = [2**14, 2**16, 2**18, 2**20]
n_sim_mc_str = ['2**14', '2**16', '2**18', '2**20'] 

discr_methods = ['massdispersal']
calculation_methods = [
    'fast_fourier_transform',
    'panjer_recursion'
]
calculation_methods_abbr = ['fft', 'recursion']

outputtxt = []

sev = Severity(par=sev_par, dist=sev_dist_name)
freq = Frequency(dist=freq_dist_name, par=freq_par, threshold=deductible)
pstr = PolicyStructure(layers=Layer(deductible=deductible, cover=cover))

calc_true_values_flag = True
for j in range(len(calculation_methods)):
    for i in range(len(n_sev_nodes)):
        print('n_sev_nodes:' + str(n_sev_nodes[i]))
        for k in range(len(discr_step)):

            print('discr_step:' + str(discr_step[k]))
            sevdict = sev.discretize(
            discr_method=discr_methods[0],
            n_discr_nodes=n_sev_nodes[i],
            discr_step=discr_step[k],
            deductible=deductible)

            lm = LossModel(
                severity=sev,
                frequency=freq,
                policystructure=pstr,
                aggr_loss_dist_method=calculation_methods_abbr[j],
                n_aggr_dist_nodes=n_aggr_nodes[i],
                n_sev_discr_nodes=n_sev_nodes[i],
                sev_discr_step=discr_step[k],
                sev_discr_method=discr_methods[0],
                tilt=True
                )
            
            if calc_true_values_flag:
                txt = [
                    'True Value & &',
                    "{:.1f}".format(lm.mean(use_dist=False)),
                    '& ', "{:.5f}".format(lm.coeff_variation(use_dist=False)),
                    '& ', "{:.5f}".format(lm.skewness(use_dist=False)),
                    '\\\\'
                    ]
                outputtxt.append(' '.join(txt))
                calc_true_values_flag = False

            vals = [
                lm.mean(use_dist=True) / lm.mean(use_dist=False) - 1,
                lm.coeff_variation(use_dist=True) / lm.coeff_variation(use_dist=False) - 1,
                lm.skewness(use_dist=True) / lm.skewness(use_dist=False) - 1
            ]
            txt = ["{:.5e}".format(x) for x in vals]
            
            if calculation_methods[j] == 'fast_fourier_transform':
                stmt = "Calculator." + calculation_methods[j]+"(" \
                    "frequency=freq," \
                    "severity=sevdict," \
                    "discr_step=" + str(discr_step[k]) +"," \
                    "tilt=True," \
                    "tilt_value=20/"+ str(n_aggr_nodes[i]) +"," \
                    "n_aggr_dist_nodes=" + str(n_aggr_nodes[i]) + "" \
                    ")"
            else:
                stmt = "Calculator." + calculation_methods[j]+"(" \
                    "frequency=freq," \
                    "severity=sevdict," \
                    "discr_step=" + str(discr_step[k]) +"," \
                    "n_aggr_dist_nodes=" + str(n_aggr_nodes[i]) + "" \
                    ")"
            out = timeit.repeat(stmt,
                        repeat=repetitions,
                        number=number,
                        globals=locals())
            txt.insert(0, str(round(np.min(out) / number, 5)))
            txt.insert(0, str(calculation_methods_abbr[j]) +' (h = ' + str(discr_step[k]) + ', m = '+ str(n_sev_nodes_str[i]) + ')')
            outputtxt.append('& '.join(txt) + '\\\\')

calc_true_values_flag = False
random_state = 11
for k in range(len(n_sim_mc)):
    lm = LossModel(
        severity=sev,
        frequency=freq,
        policystructure=pstr,
        aggr_loss_dist_method='mc',
        n_sim=n_sim_mc[k],
        random_state=random_state
        )
    if calc_true_values_flag:
        txt = [
            'True Value & &',
            "{:.1f}".format(lm.mean(use_dist=False)),
            '& ', "{:.5f}".format(lm.coeff_variation(use_dist=False)),
            '& ', "{:.5f}".format(lm.skewness(use_dist=False)),
            '\\\\'
            ]
        outputtxt.append(' '.join(txt))
        calc_true_values_flag = False
    
    vals = [
        lm.mean(use_dist=True) / lm.mean(use_dist=False) - 1,
        lm.coeff_variation(use_dist=True) / lm.coeff_variation(use_dist=False) - 1,
        lm.skewness(use_dist=True) / lm.skewness(use_dist=False) - 1
        ]
    txt = ["{:.5e}".format(x) for x in vals]

    stmt = "Calculator.mc_simulation(" \
               "severity=sev," \
               "frequency=freq," \
               "n_sim=" + str(n_sim_mc[k]) + "," \
               "random_state=" + str(random_state) + "," \
               "cover=float('inf')," \
               "deductible=" + str(deductible) + "," \
               ")"
    out = timeit.repeat(stmt,
                repeat=repetitions,
                number=number,
                globals=locals())
    txt.insert(0, str(round(np.min(out) / number, 4)))
    txt.insert(0, 'MC (' + str(n_sim_mc_str[k]) + ' sim.)')
    outputtxt.append('& '.join(txt) + '\\\\')


# export results
if explort_flag:
    pd.DataFrame(data = outputtxt, columns=['text']).to_csv(localpath + 'aggr_dist_method_comparison.txt', index=False)
    print('finish')

############################################################################################################################
# Comparison with aggregate library.
############################################################################################################################

# Parametric assumptions and contract specifications taken from examples in
# Parodi, P. (2014). Pricing in general insurance (first ed.), pag. 262-266.
sev = Severity(par=sev_par, dist=sev_dist_name)
# frequency and policystructure specified below.
outputtxt = []

n_nodes = 2**22
power_nodes = int(np.log2(n_nodes))
discr_step = 500
deductible = 10000
cover = 1000000
aggr_cover = 1000000
aggr_deductible = 50000

# No individual, no aggregate conditions
# deductible = 0
# cover = inf
# aggr. deductible = 0
# aggr. cover = inf
agg_nr = build(
    'agg noreins 3 claims sev lognorm 84541.68 cv 2.1022565437545127 poisson',
    log2=power_nodes,
    padding=0,
    bs=discr_step
    )

stmt = "build('agg noreins 3 claims sev lognorm 84541.68 cv 2.1022565437545127 poisson'," \
            "log2=" +str(power_nodes) + "," \
            "padding=0," \
            "bs=" + str(discr_step)  + "," \
            ")"
outagg = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

freq = Frequency(dist=freq_dist_name, par=freq_par, threshold=0)
pstr = PolicyStructure()
lm = LossModel(
    frequency=freq,
    severity=sev,
    policystructure=pstr,
    aggr_loss_dist_method='fft',
    n_aggr_dist_nodes=n_nodes,
    sev_discr_step=discr_step,
    sev_discr_method='massdispersal'
)
stmt = "LossModel("\
    "frequency=freq,"\
    "severity=sev,"\
    "policystructure=pstr,"\
    "aggr_loss_dist_method='fft',"\
    "n_aggr_dist_nodes=" + str(n_nodes) + "," \
    "sev_discr_step="+ str(discr_step) + "," \
    "sev_discr_method='massdispersal'"\
    ")"
outgem = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

txt = [
    ' & Reference Value & &',
    "{:.1f}".format(lm.mean(use_dist=False)),
    '& ', "{:.5f}".format(lm.coeff_variation(use_dist=False)),
    '& ', "{:.5f}".format(lm.skewness(use_dist=False)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & gemact &',
    str(round(np.min(outgem) / number, 4)) + '& ',
    "{:.1f}".format(lm.mean(use_dist=True)),
    '& ', "{:.5f}".format(lm.coeff_variation(use_dist=True)),
    '& ', "{:.5f}".format(lm.skewness(use_dist=True)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & aggregate &',
    str(round(np.min(outagg) / number, 4)) + '& ',
    "{:.1f}".format(agg_nr.est_m),
    '& ', "{:.5f}".format(agg_nr.est_cv),
    '& ', "{:.5f}".format(agg_nr.est_skew),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

# XL no aggregate conditions
# deductible = 10000
# cover = 1000000
# aggr. deductible = 0
# aggr. cover = inf
agg_xl = build(
    'agg noreins 3 claims 1000000 xs 10000 sev lognorm 84541.68 cv 2.1022565437545127 poisson',
    log2=power_nodes,
    padding=0,
    bs=discr_step
    )
agg_xl.describe

stmt = "build('agg noreins 3 claims 1000000 xs 10000 sev lognorm 84541.68 cv 2.1022565437545127 poisson'," \
            "log2=" +str(power_nodes) + "," \
            "padding=0," \
            "bs=" + str(discr_step)  + "," \
            ")"
outagg = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

pstr = PolicyStructure(
    layers=Layer(
        deductible=deductible,
        cover=cover
    )
)
freq = Frequency(dist=freq_dist_name, par=freq_par, threshold=deductible)
lm_xl = LossModel(
    frequency=freq,
    severity=sev,
    policystructure=pstr,
    aggr_loss_dist_method='fft',
    n_aggr_dist_nodes=n_nodes,
    sev_discr_method='massdispersal'
)

stmt = "LossModel("\
    "frequency=freq,"\
    "severity=sev,"\
    "policystructure=pstr,"\
    "aggr_loss_dist_method='fft',"\
    "n_aggr_dist_nodes=" + str(n_nodes) + "," \
    "sev_discr_method='massdispersal'"\
    ")"
outgem = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

txt = [
    ' & Reference Value & &',
    "{:.1f}".format(lm_xl.mean(use_dist=False)),
    '& ', "{:.5f}".format(lm_xl.coeff_variation(use_dist=False)),
    '& ', "{:.5f}".format(lm_xl.skewness(use_dist=False)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & gemact &',
    str(round(np.min(outgem) / number, 4)) + '& ',
    "{:.1f}".format(lm_xl.mean(use_dist=True)),
    '& ', "{:.5f}".format(lm_xl.coeff_variation(use_dist=True)),
    '& ', "{:.5f}".format(lm_xl.skewness(use_dist=True)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & aggregate &',
    str(round(np.min(outagg) / number, 4)) + '& ',
    "{:.1f}".format(agg_xl.est_m),
    '& ', "{:.5f}".format(agg_xl.est_cv),
    '& ', "{:.5f}".format(agg_xl.est_skew),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

# XL aggregate conditions only, i.e. SL
# deductible = 0
# cover = inf
# aggr. deductible = 50000
# aggr. cover = 1000000
agg_sl = build(
    'agg experiment 3 claims sev lognorm 84541.68 cv 2.1022565437545127 poisson aggregate ceded to 1000000 xs 50000',
    log2=power_nodes,
    padding=0,
    bs=discr_step
    )
agg_sl.describe

stmt = "build('agg experiment 3 claims sev lognorm 84541.68 cv 2.1022565437545127 poisson aggregate ceded to 1000000 xs 50000'," \
            "log2=" +str(power_nodes) + "," \
            "padding=0," \
            "bs=" + str(discr_step)  + "," \
            ")"
outagg = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

freq = Frequency(dist=freq_dist_name, par=freq_par, threshold=0)
pstr = PolicyStructure(
    layers=Layer(
        deductible = 0,
        cover = float('inf'),
        aggr_deductible = aggr_deductible,
        aggr_cover = aggr_cover
    )
)
lm_sl = LossModel(
    frequency=freq,
    severity=sev,
    policystructure=pstr,
    aggr_loss_dist_method='fft',
    n_aggr_dist_nodes=n_nodes,
    sev_discr_step=discr_step,
    sev_discr_method='massdispersal'
)

stmt = "LossModel("\
    "frequency=freq,"\
    "severity=sev,"\
    "policystructure=pstr,"\
    "aggr_loss_dist_method='fft',"\
    "n_aggr_dist_nodes=" + str(n_nodes) + "," \
    "sev_discr_step="+ str(discr_step) + "," \
    "sev_discr_method='massdispersal'"\
    ")"
outgem = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

txt = [' & Reference Value & - & - & -\\\\']
outputtxt.append(' '.join(txt))

txt = [
    ' & gemact &',
    str(round(np.min(outgem) / number, 4)) + '& ',
    "{:.1f}".format(lm_sl.mean(use_dist=True)),
    '& ', "{:.5f}".format(lm_sl.coeff_variation(use_dist=True)),
    '& ', "{:.5f}".format(lm_sl.skewness(use_dist=True)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & aggregate &',
    str(round(np.min(outagg) / number, 4)) + '& ',
    "{:.1f}".format(agg_sl.est_m),
    '& ', "{:.5f}".format(agg_sl.est_cv),
    '& ', "{:.5f}".format(agg_sl.est_skew),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

# # XL with aggregate conditions (XL + SL)
# deductible = 10000
# cover = 1000000
# aggr. deductible = 50000
# aggr. cover = 1000000
agg_xlsl = build('agg experiment 3 claims 1000000 xs 10000 sev lognorm 84541.68 cv 2.1022565437545127 poisson aggregate ceded to 1000000 xs 50000',
        log2=power_nodes,
        padding=0,
        bs=discr_step)
agg_xlsl.describe

stmt = "build('agg experiment 3 claims 1000000 xs 10000 sev lognorm 84541.68 cv 2.1022565437545127 poisson aggregate ceded to 1000000 xs 50000'," \
            "log2=" +str(power_nodes) + "," \
            "padding=0," \
            "bs=" + str(discr_step)  + "," \
            ")"
outagg = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

pstr = PolicyStructure(
    layers=Layer(
        deductible = deductible,
        cover = cover,
        aggr_deductible = aggr_deductible,
        aggr_cover = aggr_cover
    )
)
freq = Frequency(dist=freq_dist_name, par=freq_par, threshold=deductible)
lm_xlsl = LossModel(
    frequency=freq,
    severity=sev,
    policystructure=pstr,
    aggr_loss_dist_method='fft',
    n_aggr_dist_nodes=n_nodes,
    sev_discr_method='massdispersal'
)

stmt = "LossModel("\
    "frequency=freq,"\
    "severity=sev,"\
    "policystructure=pstr,"\
    "aggr_loss_dist_method='fft',"\
    "n_aggr_dist_nodes=" + str(n_nodes) + "," \
    "sev_discr_method='massdispersal'"\
    ")"
outgem = timeit.repeat(stmt,
            repeat=repetitions,
            number=number,
            globals=locals())

txt = [' & Reference Value & - & - & -\\\\']
outputtxt.append(' '.join(txt))

txt = [
    ' & gemact &',
    str(round(np.min(outgem) / number, 4)) + '& ',
    "{:.1f}".format(lm_xlsl.mean(use_dist=True)),
    '& ', "{:.5f}".format(lm_xlsl.coeff_variation(use_dist=True)),
    '& ', "{:.5f}".format(lm_xlsl.skewness(use_dist=True)),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

txt = [
    ' & aggregate &',
    str(round(np.min(outagg) / number, 4)) + '& ',
    "{:.1f}".format(agg_xlsl.est_m),
    '& ', "{:.5f}".format(agg_xlsl.est_cv),
    '& ', "{:.5f}".format(agg_xlsl.est_skew),
    '\\\\'
    ]
outputtxt.append(' '.join(txt))

# export results
if explort_flag:
    pd.DataFrame(data = outputtxt, columns=['text']).to_csv(localpath + 'aggregate_comparison.txt', index=False)
