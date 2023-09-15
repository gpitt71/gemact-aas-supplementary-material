# gemact-aas-supplementary-material
Supplementary material of GEMAct: a Python package for non-life (re)insurance modelling.

* `paper_code_blocks.py` contains the code blocks of the manuscript.
* `lossmodel_comparisons.py` contains the code for the internal comparison of the methods to
compute the aggregate loss distribution (Table 1), and the comparison of FFT implementa-
tions with aggregate library (Table 2)
* `lossaggregation_accuracy_speed.py` contains the code for the comparison of the AEP and
MC methods to compute the cdf (Table 3).
* `lossaggregation_sensitivity.py` contains the code for the sensitivity analysis of the AEP
and MC methods to compute the cdf (Table 4).
* `lossreserve_comparison_w_chl.py` contains the code to compute the CRMR results.
* `reserving_data_generator_and_chl.R` contains the code to generate the simulated data set
that we save in the module gemdata and the code to compute Mack’s Chain Ladder.
