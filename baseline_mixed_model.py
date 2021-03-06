
import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
import statsmodels.api as sm
from phenotype_correlation import *
from sklearn.linear_model import Ridge 

"""
baseline mixed model 

Y = ZA + XB + E 

we calculate the ZA using OLS first
and perform ridge on the residuals (Y - ZA ~ N(XB, sigma)


"""
def baseline_mixed_model_analysis(geno_df, pheno_df, phenotype_1, phenotype_2, missing_rate = 0.1, sample_list = list(), verbose = False):

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
		print(lm.params)	

	# prediction for fixed effect
	predictions_fe = lm.predict(pheno_test[phenotype_1])

	# perform ridge regression on the residual (random effect part)
	residuals = pheno_tr[phenotype_2] - lm.predict(pheno_tr[phenotype_1])

	lm_re = sm.OLS(endog = residuals, exog = geno_tr.transpose()).fit_regularized(L1_wt = 1.0)

	if verbose: 
		print(lm_re.params)

	predictions_re = lm_re.predict(geno_test.transpose())

	# combine the result from both
	total_prediction = predictions_fe + predictions_re

	mse = calculate_MSE(total_prediction, pheno_test[phenotype_2])

	return(mse, test_sample_list)



def top_N_snp_mixed_model_analysis(geno_df, pheno_df, phenotype_1, phenotype_2, top_N = 100, missing_rate = 0.1, sample_list = list(), verbose = False):

	corr_mat = calculate_highly_correlated_phenotypes(pheno_df)

	print("The correlation between %s and %s is %f" % (phenotype_1, phenotype_2, corr_mat[phenotype_1][phenotype_2]))

	# bind phenotype into list to extract
	phenotype_list = [phenotype_1, phenotype_2]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_multiple_phenotypes(geno_df, pheno_df, phenotype_list = phenotype_list, verbose = verbose)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test, test_sample_list = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate, sample_list_select = sample_list)

	# remove duplciates
	geno_test_new = geno_test.loc[:,~geno_test.columns.duplicated()]
	geno_test = geno_test_new[pheno_test[phenotype_2].index]

	# saving below 
	# # perform simple ridge to identify the top SNPs 
	# lm_ridge = sm.OLS(endog = pheno_tr[phenotype_2], exog = geno_tr.transpose()).fit_regularized(L1_wt = 1.0)

	# if verbose: 
	#     print(lm_ridge.params)

	# # select top SNPs with highest effect size for select run
	# top_N_idx = np.argsort(abs(lm_ridge.params))[-top_N:]

	# if verbose:
	# 	top_N_values = [lm_re.params[i] for i in top_N_idx]
	# 	print(top_N_values)

	# top_N_snps = geno_tr.iloc[top_N_idx].index

	# sklearn test 

	# clf = Ridge(alpha = 1.0)
	# a = clf.fit(y = pheno_tr[phenotype_2], X = geno_tr.transpose())

	# # select top N 
	# top_N = 10
	# top_N_idx = np.argsort(abs(a.coef_))[-top_N:]

	# print (top_N_idx)

	# top_N_values = [a.coef_[i] for i in top_N_idx]
	# print (top_N_values)

	# top_N_snps = geno_tr.iloc[top_N_idx].index
	# print(top_N_snps)




	# perform OLS 
	lm = sm.OLS(endog = pheno_tr[phenotype_2], exog = pheno_tr[phenotype_1]).fit()
	
	if verbose:	
		print("The linear model summary for predicting phenotype %a based on phenotype %a" % (phenotype_2, phenotype_1))
		print(lm.summary())	
		print(lm.params)	

	# prediction for fixed effect
	predictions_fe = lm.predict(pheno_test[phenotype_1])

	# perform ridge regression on the residual (random effect part)
	residuals = pheno_tr[phenotype_2] - lm.predict(pheno_tr[phenotype_1])

	# check marginal 

	num_SNPs = geno_tr.shape[0]
	beta_list = []

	for snp_idx in range(num_SNPs):
		lm_snp = sm.OLS(endog = residuals, exog = geno_tr.iloc[snp_idx].transpose()).fit_regularized(L1_wt = 1.0, alpha = 1.0)

	#     clf = Ridge(alpha = 1.0)
	#     a = clf.fit(y = residuals, X = geno_tr.iloc[snp_idx].transpose())
		beta_list.append(lm_snp.params)

		if snp_idx % 1000 == 0: 
			print(snp_idx)
	        
	beta = pd.concat(beta_list)

	top_N_idx = np.argsort(abs(beta))[-top_N:]

	top_N_values = [beta[i] for i in top_N_idx]

	top_N_snps = geno_tr.iloc[top_N_idx].index



	lm_re = sm.OLS(endog = residuals, exog = geno_tr.loc[top_N_snps].transpose()).fit_regularized(L1_wt = 1.0, alpha = 1.0)

	if verbose: 
		print(lm_re.params)

	predictions_re = lm_re.predict(geno_test.loc[top_N_snps].transpose())

	# combine the result from both
	total_prediction = predictions_fe + predictions_re

	print (predictions_re, predictions_fe)

	mse = calculate_MSE(total_prediction, pheno_test[phenotype_2])

	return(mse, test_sample_list)


def top_N_snp_mixed_model_analysis_p(geno_df, pheno_df, phenotype_1, phenotype_2, top_N = 100, missing_rate = 0.1, sample_list = list(), verbose = False):

	corr_mat = calculate_highly_correlated_phenotypes(pheno_df)

	print("The correlation between %s and %s is %f" % (phenotype_1, phenotype_2, corr_mat[phenotype_1][phenotype_2]))

	# bind phenotype into list to extract
	phenotype_list = [phenotype_1, phenotype_2]

	# extract the phenotypes 
	geno_select, pheno_select = select_phenotype_multiple_phenotypes(geno_df, pheno_df, phenotype_list = phenotype_list, verbose = verbose)

	# separate training and test dataset 
	geno_tr, pheno_tr, geno_test, pheno_test, test_sample_list = separate_training_test(geno_select, pheno_select, missing_rate = missing_rate, sample_list_select = sample_list)

	# remove duplciates
	geno_test_new = geno_test.loc[:,~geno_test.columns.duplicated()]
	geno_test = geno_test_new[pheno_test[phenotype_2].index]

	# perform OLS 
	lm = sm.OLS(endog = pheno_tr[phenotype_2], exog = pheno_tr[phenotype_1]).fit()
	
	if verbose:	
		print("The linear model summary for predicting phenotype %a based on phenotype %a" % (phenotype_2, phenotype_1))
		print(lm.summary())	
		print(lm.params)	

	# prediction for fixed effect
	predictions_fe = lm.predict(pheno_test[phenotype_1])

	# perform ridge regression on the residual (random effect part)
	residuals = pheno_tr[phenotype_2] - lm.predict(pheno_tr[phenotype_1])

	# check marginal 

	num_SNPs = geno_tr.shape[0]

	beta_list = []
	p_beta_list = []
	for snp_idx in range(num_SNPs):
			lm_snp = sm.OLS(endog = residuals, exog = geno_tr.iloc[snp_idx].transpose()).fit()

			p_val = lm_snp.pvalues[0]
			beta = lm_snp.params[0]
			if p_val < 0.05:
				beta_list.append(beta)
				p_beta_list.append(pd.Series([beta, p_val], name = geno_tr.iloc[snp_idx].name))
			if snp_idx % 1000 == 0: 
				print(snp_idx)

	p_beta_df = pd.concat(p_beta_list, axis = 1).transpose()
	p_beta_df.columns = ["beta", "pval"]

	p_beta_df.sort_values(by = ['pval'], inplace = True)

	top_N = min(top_N, p_beta_df.shape[0])

	top_N_snps = p_beta_df.iloc[range(top_N)].index



	lm_re = sm.OLS(endog = residuals, exog = geno_tr.loc[top_N_snps].transpose()).fit_regularized(L1_wt = 1.0, alpha = 1.0)

	if verbose: 
		print(lm_re.params)

	predictions_re = lm_re.predict(geno_test.loc[top_N_snps].transpose())

	# combine the result from both
	total_prediction = predictions_fe + predictions_re

	print (predictions_re, predictions_fe)

	mse = calculate_MSE(total_prediction, pheno_test[phenotype_2])

	return(mse, test_sample_list, top_N)