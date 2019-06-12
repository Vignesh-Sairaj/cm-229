import sys
import os 
import pandas as pd
import numpy as np


"""
IMPORT DATA FOR THE MODEL 
RETURN: GENOTYPE_DF, PHENOTYPE_DF 
"""
def import_data_islets():

	input_dir = "./DO_Islets/"
	# import genotype
	input_geno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_geno.csv" % (input_dir)

	input_geno_df = pd.read_csv(input_geno_file, index_col = 0)

	geno_cols = input_geno_df.columns.tolist()

	# import phenotype
	input_pheno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_pheno.csv" % (input_dir)

	input_pheno_df = pd.read_csv(input_pheno_file, index_col = 0)

	pheno_samples = input_pheno_df.index.tolist()
	
	# generate sample list
	sample_list = [ x for x in geno_cols if x in pheno_samples ]

	print("Samples:")
	print(sample_list)

	# select the genotypes for the samples 
	geno_df_select = input_geno_df[sample_list]

	# select the phenotypes for the samples 
	pheno_df_select = input_pheno_df.loc[sample_list]

	return geno_df_select, pheno_df_select 

def import_data_svenson():

	input_dir = "./svenson/"
	# import genotype
	input_geno_file = "%s/Svenson-183_Svenson_DO-MegaMUGA_geno.csv" % (input_dir)

	input_geno_df = pd.read_csv(input_geno_file, index_col = 0)

	geno_cols = input_geno_df.columns.tolist()

	# import phenotype
	input_pheno_file = "%s/Svenson-183_Svenson_DO-MegaMUGA_pheno.csv" % (input_dir)

	input_pheno_df = pd.read_csv(input_pheno_file, index_col = 0)

	pheno_samples = input_pheno_df.index.tolist()
	
	# generate sample list
	sample_list = [ x for x in geno_cols if x in pheno_samples ]

	print("Samples:")
	print(sample_list)

	# select the genotypes for the samples 
	geno_df_select = input_geno_df[sample_list]

	# select the phenotypes for the samples 
	pheno_df_select = input_pheno_df.loc[sample_list]

	return geno_df_select, pheno_df_select 


def select_phenotype_islets(geno_df, pheno_df, phenotype = "num_islets"):
	# select a phenotype for genotype baseline model 
	# we are going to use num_islets as the test phenotype for baseline model
	phenotype = pheno_df[phenotype]

	# ID samples with missing phenotype
	samples_with_missing_pheno = phenotype[phenotype < 0].index.tolist()

	print("These samples are missing phenotypes:")
	print(samples_with_missing_pheno)

	# select samples with known phenotypes
	phenotype_complete = pd.DataFrame(phenotype[~phenotype.index.isin(samples_with_missing_pheno)])

	# save the complete_sample_list for identifying matching genotype
	complete_sample_list = phenotype_complete.index.tolist()

	# select genotype
	genotype_complete = geno_df[complete_sample_list]

	print(genotype_complete.shape)

	# clean snps with missing geno 
	genotype_complete_dropNA = genotype_complete.dropna()

	print(genotype_complete_dropNA.shape)
	print(phenotype_complete.shape)


	return genotype_complete_dropNA, phenotype_complete

def select_phenotype_multiple_phenotypes(geno_df, pheno_df, phenotype_list):
	# select a phenotype for genotype baseline model 
	# we are going to use num_islets as the test phenotype for baseline model
	phenotype = pheno_df[phenotype_list]

	# ID samples with missing phenotype
	samples_with_missing_pheno = phenotype[phenotype < -99999].dropna(how = "any").index.tolist()

	print("These samples are missing phenotypes:")
	print(samples_with_missing_pheno)

	# select samples with known phenotypes
	phenotype_complete = pd.DataFrame(phenotype[~phenotype.index.isin(samples_with_missing_pheno)])

	# save the complete_sample_list for identifying matching genotype
	complete_sample_list = phenotype_complete.index.tolist()

	# select genotype
	genotype_complete = geno_df[complete_sample_list]

	print(genotype_complete.shape)

	# clean snps with missing geno 
	genotype_complete_dropNA = genotype_complete.dropna()

	print(genotype_complete_dropNA.shape)
	print(phenotype_complete.shape)


	return genotype_complete_dropNA, phenotype_complete

def select_phenotype_single_phenotype(geno_df, pheno_df, phenotype_list):
	# select a phenotype for genotype baseline model 
	# we are going to use num_islets as the test phenotype for baseline model
	phenotype = pheno_df[phenotype_list]

	# ID samples with missing phenotype
	samples_with_missing_pheno = phenotype[phenotype < -99999].dropna(how = "any").index.tolist()

	print("These samples are missing phenotypes:")
	print(samples_with_missing_pheno)

	# select samples with known phenotypes
	phenotype_complete = pd.DataFrame(phenotype[~phenotype.index.isin(samples_with_missing_pheno)])

	# save the complete_sample_list for identifying matching genotype
	complete_sample_list = phenotype_complete.index.tolist()

	# select genotype
	genotype_complete = geno_df[complete_sample_list]

	print(genotype_complete.shape)

	# clean snps with missing geno 
	genotype_complete_dropNA = genotype_complete.dropna()

	print(genotype_complete_dropNA.shape)
	print(phenotype_complete.shape)


	return genotype_complete_dropNA, phenotype_complete



def separate_training_test(geno_df, pheno_df, missing_rate = 0.1, sample_list_select = list()):

	sample_list = geno_df.columns.to_list()

	missing_num = int(missing_rate * geno_df.shape[1])

	if len(sample_list_select) == 0: # if sample list is NOT give, then select samples randomly
		sample_list_testing = np.random.choice(sample_list, int(missing_num))
	else: # if sample list is given 
		sample_list_testing = sample_list_select


	sample_list_training = [ x for x in sample_list if x not in sample_list_testing ]
	
	# separate the training and test 
	phenotype_training = pd.DataFrame(pheno_df[~pheno_df.index.isin(sample_list_testing)])
	phenotype_testing = pd.DataFrame(pheno_df[pheno_df.index.isin(sample_list_testing)])

	genotype_testing = geno_df[sample_list_testing]
	genotype_training = geno_df[sample_list_training]

	return genotype_training, phenotype_training, genotype_testing, phenotype_testing, sample_list_testing



if __name__ == "__main__":
	test()