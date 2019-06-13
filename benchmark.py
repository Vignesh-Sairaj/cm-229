import os
import sys
import pandas as pd 
import numpy as np
from data_import import *
from phenotype_correlation import *
from genotype_correlation import *
from baseline_mixed_model import *


# import data
geno_df, pheno_df = import_data_svenson()


##  actual benchmarking run 
# testing with high correlation weight1 and length1 
# initialization for the benchmark run
phenotype_1 = "weight1"
phenotype_2 = "length1"

missing_rate = float(sys.argv[1])
print(missing_rate)

num_runs = 10

result_list = [] 


for i in range(num_runs):
    
    print("Running missing rate %f run %i" % (missing_rate, i))

    mse, test_sample_list = phenotype_correlation_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate)
    result_pheno = pd.DataFrame([missing_rate, i, mse, "Phenotype_correlation"])
    result_list.append(result_pheno)

    genotype_ridge_mse, _sample_list = genotype_correlation_analysis_ridge(geno_df, pheno_df, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list)
    result_geno = pd.DataFrame([missing_rate, i, genotype_ridge_mse, "Genotype_correlation"])
    result_list.append(result_geno)
    
    mm_mse, _sample_list = baseline_mixed_model_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list)
    result_mm = pd.DataFrame([missing_rate, i, mm_mse, "Baseline_mixed_model"])
    result_list.append(result_mm)
    
    mm_mse, _sample_list = top_N_snp_mixed_model_analysis(geno_df, pheno_df, phenotype_2, phenotype_1, missing_rate = missing_rate, sample_list = test_sample_list, top_N = 10)
    result_mm = pd.DataFrame([missing_rate, i, mm_mse, "top_N_mixed_model"])
    result_list.append(result_mm)


result_df = pd.concat(result_list, axis = 1).transpose()
result_df.columns = ["missing_rate", "run", "MSE", "method"]


result_df.to_csv('./result.%s.csv' % (str(missing_rate)), index = False)

