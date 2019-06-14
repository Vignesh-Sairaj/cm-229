import numpy as np
from scipy.stats import matrix_normal, wishart
from statsmodels.stats.moment_helpers import cov2corr


def generate_B(P=15, rho=0.45):
	'''
	Generate the P x P AR(1) matrix with rho = 0.45
	'''
	return np.fromiter((rho**abs(i-j) for i in range(P) for j in range(P)), dtype=float, count=P*P).reshape((P, P))

def generate_E(P=15):
	return wishart.rvs(df=P, scale=(1.0/P)*np.eye(P))

def generate_pheno(kinship, hsquared, N=300, P=15, rho=0.45):
	'''
	Generates phenotype data from MN distribution
	N = n_samples, P = n_traits, and rho is the autocorrelation parameter to B
	kinship matrix must be NxN
	RETURNS ndarray of size (N x P)
	'''

	assert kinship.shape == (N, N)
	U = matrix_normal.rvs( rowcov=kinship, colcov=hsquared*cov2corr(generate_B(P=15, rho=0.45)) )
	epsilon = matrix_normal.rvs( rowcov=np.eye(N), colcov=(1-hsquared)*cov2corr(generate_E(P=15)) )

	return U + epsilon