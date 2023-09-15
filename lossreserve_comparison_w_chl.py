from gemact import gemdata
from gemact import AggregateData, ReservingModel, LossReserve
import numpy as np

# Load the data: the data is simulated using the R code available at https://gem-analytics.github.io/gemact/gemdata.html.
# The R script is provided in a separate file.

ip = gemdata.incremental_payments_sim
pnb = gemdata.payments_number_sim
cp = gemdata.cased_payments_sim
opn = gemdata.open_number_sim
reported = gemdata.reported_claims_sim
czj = gemdata.czj_sim


############################################################################################################################
# Computation of the CRM and the Mack CL
############################################################################################################################

ad = AggregateData(
    incremental_payments=ip,
    cased_payments=cp,
    open_claims_number=opn,
    reported_claims=reported,
    payments_number=pnb)

resmodel_crm = ReservingModel(
    tail=False,
    reserving_method='crm',
    claims_inflation=np.array([1]),
    mixing_fq_par=.01,
    mixing_sev_par=.01,
    czj= czj)

lr = LossReserve(
    data=ad,
    reservingmodel=resmodel_crm,
    ntr_sim=1000,
    random_state=42)

lr.reserve
lr.ppf(q=np.array([.25,.5,.75,.995,.9995]))/10**6

# Table 5: Reserves by accident period for the CRMR. Amounts are reported in millions

outputtxt=[]
txt = [' Accident Period & CRMR reserve & CRMR msep \\\\']
outputtxt.append(' '.join(txt))

ap = np.arange(0, 9)

for i in range(0, 9):
    txt = [
        str(ap[i]) + '& ',
        "{:.2f}".format(np.round(lr.crm_reserve_ay[i]/10**6,2)),
        '& ', "{:.2f}".format(np.round(lr.crm_sep_ay[i]/10**6, 2)),
        '\\\\'
    ]
    outputtxt.append(' '.join(txt))

txt = [
        'Total' + '& ',
        "{:.2f}".format(np.round(lr.reserve/10**6,2)),
        '& ', "{:.2f}".format(np.round(lr.m_sep/10**6, 2)),
        '\\\\'
    ]
outputtxt.append(' '.join(txt))

# Table 6: Total reserve estimates, their relative value, as a fraction of the actual value (8599.04),
# and their coefficient of variations (CoV), for the CRMR and the CHL. Absolute amounts are reported in millions.

outputtxt=[]
txt = [' Reserve & Reserve/Actual & CoV \\\\']
outputtxt.append(' '.join(txt))

txt = [
        'CRMR' + '& ',
        "{:.2f}".format(np.round(lr.reserve/np.sum(ip[lr.data.ix > lr.data.j]),2)),
        '& ', "{:.2f}".format(np.round(100*lr.m_sep/lr.reserve, 2)),
        '\\\\'
    ]
outputtxt.append(' '.join(txt))