
import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
import statsmodels.api as sm
from phenotype_correlation import *

"""
baseline mixed model 

Y = ZA + XB + E 

we calculate the ZA using OLS first
and perform ridge on the residuals (Y - ZA ~ N(XB, sigma)


"""
def baseline_mixed_model_analysis(geno_df, pheno_df, phenotype_1, phenotype_2, missing_rate = 0.1, sample_list = list()):

	corr_mat = calculate_highly_correlated_phenotypes(pheno_df)

	print("The correlation between %s and %s is %f" % (phenotype_1, phenotype_2, corr_mat[phenotype_1][phenotype_2]))

	# bind phenotype into list to extract
	phenotype_list = [phenotype_1, phenotype_2]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_multiple_phenotypes(geno_df, pheno_df, phenotype_list = phenotype_list)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test, test_sample_list = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate, sample_list_select = sample_list)

	# perform OLS 
	lm = sm.OLS(endog = pheno_tr[phenotype_2], exog = pheno_tr[phenotype_1]).fit()
	
	print("The linear model summary for predicting phenotype %a based on phenotype %a" % (phenotype_2, phenotype_1))
	print(lm.summary())

	# prediction for fixed effect
	predictions_fe = lm.predict(pheno_test[phenotype_1])

	# perform ridge regression on the residual (random effect part)
	residuals = pheno_tr[phenotype_2] - lm.predict(pheno_tr[phenotype_1])

	lm_re = sm.OLS(endog = residuals, exog = geno_tr.transpose()).fit_regularized()

	print(lm_re.summary())

	predictions_re = lm_re.predict(geno_test.transpose())

	# combine the result from both
	total_prediction = predictions_fe + predictions_re

	mse = calculate_MSE(total_prediction, pheno_test[phenotype_2])

	return(mse, test_sample_list)