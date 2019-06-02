import sys
import os 
import pandas as pd


def test(): 
	input_dir = "./DO_Islets/"

	# import genotype
	input_geno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_geno.csv" % (input_dir)

	input_geno_df = pd.read_csv(input_geno_file, nrows = 1000, index_col = 0)

	geno_cols = input_geno_df.columns.tolist()	

	# import phenotype
	input_pheno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_pheno.csv" % (input_dir)

	input_pheno_df = pd.read_csv(input_pheno_file, index_col = 0)

	pheno_samples = input_pheno_df.index.tolist()

	print [ x for x in geno_cols if x in pheno_samples]

if __name__ == "__main__":
	test()