from gemact import Margins, Copula
from gemact.calculators import MCCalculator, AEPCalculator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local path to save
localpath = ''
explort_flag = False

# array to store outputs
copula1 = 'gaussian'
copula2 = 'clayton'
container = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

# values where the cdf is evaluated in 2d and 3d
sValues = np.array([[1.25, 1.5, 1.75, 1.95],
                    [2, 2.3, 2.65, 2.85]])
nsim = 10**7

# create Margins
margins = Margins(dist=['uniform', 'uniform'], par=[{'a': 0, 'b': 1}, {'a': 0, 'b': 1}])
margins3 = Margins(dist=['uniform', 'uniform', 'uniform'],
                   par=[{'a': 0, 'b': 1}, {'a': 0, 'b': 1}, {'a': 0, 'b': 1}])


#######################################################################################
# Gaussian copulas
#######################################################################################
# create Copula objects
# correlation values
correlations = np.linspace(-0.95, 0.95, num=19, endpoint=True)
copulas = [
    Copula(dist='gaussian',
           par={'corr': np.array([[1, corr], [corr, 1]])}
           )
    for corr in correlations
    ]

for i in range(len(copulas)):
    print(i)
    cop = copulas[i]
    p_aep = AEPCalculator.cdf(sValues[0, ], n_iter=7, copula=cop, margins=margins)
    container[0][copula1 + ': ' + str(correlations[i])] = p_aep

    simul = MCCalculator.rvs(size=nsim, random_state=55+i, copula=cop, margins=margins).reshape(-1, 1)
    simul = simul < sValues[0, ]
    p_mc = np.mean(simul, axis=0)
    container[1][copula1 + ': ' + str(correlations[i])] = p_mc
        
# export results
if explort_flag == True:
    container[0].to_csv(str(localpath + '\AEP_sens_' + copula1 + '.csv'), sep=",", index=False)
    container[1].to_csv(str(localpath + '\MC_sens_' + copula1 + '.csv'), sep=",", index=False)
    # save as txt
    # np.savetxt(str(localpath + '\AEP_sens_' + copula1 + '.txt'), container[0].values)
    # np.savetxt(str(localpath + '\MC_sens_' + copula1 + '.txt'), container[1].values)

#######################################################################################
# Clayton copulas
#######################################################################################
# create Copula objects
thetas = np.linspace(0.05, 20, num=19, endpoint=True)
copulas = [
    Copula(dist='clayton',
           par={'par': theta, 'dim': 3}
           )
    for theta in thetas
    ]

for i in range(len(copulas)):
    print(i)
    cop = copulas[i]
    p_aep = AEPCalculator.cdf(sValues[1, ], n_iter=7, copula=cop, margins=margins3)
    container[2][copula2 + ': ' + str(thetas[i])] = p_aep

    simul = MCCalculator.rvs(size=nsim, random_state=55+i, copula=cop, margins=margins3).reshape(-1, 1)
    simul = simul < sValues[1, ]
    p_mc = np.mean(simul, axis=0)
    container[3][copula2 + ': ' + str(thetas[i])] = p_mc
        
# export results
if explort_flag == True:
    container[2].to_csv(str(localpath + '\AEP_sens_' + copula2 + '.csv'), sep=",", index=False)
    container[3].to_csv(str(localpath + '\MC_sens_' + copula2 + '.csv'), sep=",", index=False)
    # save as txt
    # np.savetxt(str(localpath + '\AEP_sens_' + copula2 + '.txt'), container[2].values)
    # np.savetxt(str(localpath + '\MC_sens_' + copula2 + '.txt'), container[3].values)

# plot
coloraep = 'black'
colormc = 'pink'
colorerror = 'red'
colortaberror = 'tab:red'
limiterror = [0, 0.0005]
linestylemc = 'dashed'
linestyleerror = 'dotted'
fontsize_title= 15
fontsize_axis = 15
linewidth_aep = 1
pos_text_y = [[0.25, 0.55, 0.85, 0.96],
              [0.08, 0.35, 0.70, 0.90]]
correlations = np.linspace(-1, 1, num=19, endpoint=True) 

fig1 = plt.figure(figsize=(8, 3))

ax1 = fig1.add_subplot(121)
ax1.set_xlabel(r'$\rho$', fontsize=fontsize_axis)
ax1.set_ylabel('cdf', fontsize=fontsize_axis)
ax1.set_title('Gaussian Copula', multialignment='center', fontsize=fontsize_title)
ax1.tick_params(axis='both', which='major', labelsize=15)
for j in range(sValues.shape[1]):
    if j == 0:
        ax1.plot(correlations, container[0].iloc[j].values,
                 color=coloraep,
                 linewidth=linewidth_aep,
                 label="AEP")
    else:
        ax1.plot(correlations, container[0].iloc[j].values,
                linewidth=linewidth_aep,
                color=coloraep
                )
    plt.text(0.75, pos_text_y[0][j],
             's = ' + str(sValues[0, j]),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax1.transAxes,
             fontsize='large')

ax3 = ax1.twinx()  
ax3.plot(correlations,
         np.abs(container[0] - container[1]).mean(0).values,
         color=colorerror,
         linestyle=linestyleerror,
         label="|AEP - MC|")
ax3.set_ylabel('|AEP - MC|', color=colortaberror, fontsize=fontsize_axis)
ax3.set_ylim(limiterror)
ax3.tick_params(axis='y', labelcolor=colortaberror, labelsize=15)

ax2 = fig1.add_subplot(122)
ax2.set_xlabel(r'$\theta$', fontsize=fontsize_axis)
ax2.set_ylabel('cdf', fontsize=fontsize_axis)
ax2.set_title('Clayton Copula', multialignment='center', fontsize=fontsize_title)
ax2.tick_params(axis='both', which='major', labelsize=15)
for j in range(sValues.shape[1]):
    if j == 0:
        ax2.plot(thetas, container[2].iloc[j].values,
                color=coloraep,
                linewidth=linewidth_aep,
                )
    else:
        ax2.plot(thetas, container[2].iloc[j].values,
                linewidth=linewidth_aep,
                color=coloraep
                )
    plt.text(0.75, pos_text_y[1][j],
             's = ' + str(sValues[1, j]),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax2.transAxes,
             fontsize='large')
ax4 = ax2.twinx()  
ax4.plot(
    thetas,
    np.abs(container[2] - container[3]).mean(0).values,
    color=colorerror,
    linestyle=linestyleerror,
    )
ax4.set_ylabel('|AEP - MC|', fontsize=fontsize_axis, color=colortaberror)
ax4.set_ylim(limiterror)
ax4.tick_params(axis='y', labelcolor=colortaberror, labelsize=15)
fig1.savefig(str(localpath + '\lossaggr_sens_copula2.eps'))
