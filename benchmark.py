import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
from phenotype_correlation import *
from genotype_correlation import *
from baseline_mixed_model import *
import argparse

# import data
geno_df, pheno_df = import_data_svenson()


##  actual benchmarking run 
# testing with high correlation weight1 and length1 
# initialization for the benchmark run
# phenotype_1 = "weight1"
# phenotype_2 = "length1"

parser = argparse.ArgumentParser()
parser.add_argument("missing_rate", action = "store")
parser.add_argument("phenotype_to_predict", action = "store")
parser.add_argument("correlated_phenotype", action = "store")

parser.add_argument("--num_SNPs", help = "number of Top SNPS to use for the model", 
    default = 10)

parser.add_argument("--model_test", help = "test our model only, not others", 
    action = "store_true")

parser.add_argument("--num_runs", help = "number of runs to test ", default = 10)

args = parser.parse_args()


### Parse arguments ### 
missing_rate = float(args.missing_rate)
phenotype_1 = str(args.phenotype_to_predict)
phenotype_2 = str(args.correlated_phenotype)
num_SNPs = int(args.num_SNPs)
num_runs = int(args.num_runs)
print(missing_rate, phenotype_1, phenotype_2)
model_test_mode = args.model_test


result_list = [] 

if model_test_mode:
    print("testing model")
    print(num_SNPs)

for i in range(num_runs):
    
    print("Running missing rate %f run %i" % (missing_rate, i))

    if not model_test_mode:
        mse, test_sample_list = phenotype_correlation_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate)
        result_pheno = pd.DataFrame([missing_rate, i, mse, "Phenotype_correlation"])
        result_list.append(result_pheno)

        genotype_ridge_mse, _sample_list = genotype_correlation_analysis_ridge(geno_df, pheno_df, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list)
        result_geno = pd.DataFrame([missing_rate, i, genotype_ridge_mse, "Genotype_correlation"])
        result_list.append(result_geno)
        
        mm_mse, _sample_list = baseline_mixed_model_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list)
        result_mm = pd.DataFrame([missing_rate, i, mm_mse, "Baseline_mixed_model"])
        result_list.append(result_mm)
        
    mm_top_mse, _sample_list = top_N_snp_mixed_model_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list, top_N = num_SNPs)
    result_mm_top = pd.DataFrame([missing_rate, i, mm_top_mse, "top_N_mixed_model"])
    result_list.append(result_mm_top)


result_df = pd.concat(result_list, axis = 1).transpose()
result_df.columns = ["missing_rate", "run", "MSE", "method"]

if model_test_mode:
    print(result_df)

if not model_test_mode:
    result_df.to_csv('./result/result.%s.%s_%s.csv' % (str(missing_rate), phenotype_1, phenotype_2), index = False)
if model_test_mode:
    result_df.to_csv('./result/result.%s.%s_%s.%s.csv' % (str(missing_rate), phenotype_1, phenotype_2, str(num_SNPs)), index = False)
