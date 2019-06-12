
import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
import statsmodels.api as sm
from phenotype_correlation import calculate_MSE

def genotype_correlation_analysis(geno_df, pheno_df, phenotype, missing_rate = 0.1):

	# bind phenotype into list to extract
	phenotype_list = [phenotype]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_single_phenotype(geno_df, pheno_df, phenotype_list = phenotype_list)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate)

	# perform OLS
	lm = sm.OLS(endog = pheno_tr[phenotype], exog = geno_tr.transpose()).fit()

	print("The linear model summary for predicting phenotype %a based on genotype" % (phenotype))
	print(lm.summary())

	predictions = lm.predict(geno_test.transpose())

	mse = calculate_MSE(predictions, pheno_test[phenotype])

	return(mse)

def genotype_correlation_analysis_ridge(geno_df, pheno_df, phenotype, missing_rate = 0.1):

	# bind phenotype into list to extract
	phenotype_list = [phenotype]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_single_phenotype(geno_df, pheno_df, phenotype_list = phenotype_list)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate)

	# perform OLS
	lm = sm.OLS(endog = pheno_tr[phenotype], exog = geno_tr.transpose()).fit_regularized()

	print("The linear model summary for predicting phenotype %a based on genotype" % (phenotype))
	print(lm.summary())

	predictions = lm.predict(geno_test.transpose())

	mse = calculate_MSE(predictions, pheno_test[phenotype])

	return(mse)