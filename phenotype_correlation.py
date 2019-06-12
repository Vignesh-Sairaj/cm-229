
# coding: utf-8

import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
import statsmodels.api as sm


"""
Calculate the correlation between phenotypes and select
"""
def calculate_highly_correlated_phenotypes(pheno_df, low_threshold = 0.5, high_threshold = 1):
	corr_mat = pheno_df.corr()
	high_cor = corr_mat[(corr_mat.abs() > low_threshold) & (corr_mat.abs() < high_threshold)].dropna(how = "all")

	return(high_cor)

"""
perform phenotype correlation analysis
Phenotype 1 is used to predict phenotype 2
"""
def phenotype_correlation_analysis(geno_df, pheno_df, phenotype_1, phenotype_2, missing_rate = 0.1, sample_list = list(), verbose = False):

	corr_mat = calculate_highly_correlated_phenotypes(pheno_df)

	print("The correlation between %s and %s is %f" % (phenotype_1, phenotype_2, corr_mat[phenotype_1][phenotype_2]))

	# bind phenotype into list to extract
	phenotype_list = [phenotype_1, phenotype_2]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_multiple_phenotypes(geno_df, pheno_df, phenotype_list = phenotype_list, verbose = verbose)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test, test_sample_list = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate, sample_list_select = sample_list)

	# perform OLS 
	lm = sm.OLS(endog = pheno_tr[phenotype_2], exog = pheno_tr[phenotype_1]).fit()
	
	if verbose:
		print("The linear model summary for predicting phenotype %a based on phenotype %a" % (phenotype_2, phenotype_1))
		print(lm.summary())

	predictions = lm.predict(pheno_test[phenotype_1])

	mse = calculate_MSE(predictions, pheno_test[phenotype_2])

	return(mse, test_sample_list)


def generate_correlation_plot(high_cor):
	import seaborn as sns
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(20, 20))

	heatmap = sns.heatmap(high_cor, ax = ax)
	heatmap.figure

def calculate_MSE(prediction, actual):
	error = prediction - actual
	mse = sum(error ** 2) / len(prediction)
	return(mse)

