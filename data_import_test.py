
# coding: utf-8

# In[3]:


import sys
import os 
import pandas as pd


# In[4]:


input_dir = "./DO_Islets/"

# import genotype
input_geno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_geno.csv" % (input_dir)

input_geno_df = pd.read_csv(input_geno_file, index_col = 0)

geno_cols = input_geno_df.columns.tolist()

# import phenotype
input_pheno_file = "%s/Attie-232_Attie_DO_Islets-GigaMUGA_pheno.csv" % (input_dir)

input_pheno_df = pd.read_csv(input_pheno_file, index_col = 0)

pheno_samples = input_pheno_df.index.tolist()


# In[6]:


# generate sample list
sample_list = [ x for x in geno_cols if x in pheno_samples ]


# In[7]:


print(sample_list)


# In[8]:


# select the genotypes for the samples 
geno_df_select = input_geno_df[sample_list]


# In[12]:


# select the phenotypes for the samples 
pheno_df_select = input_pheno_df.loc[sample_list]


# In[14]:


# check the dimensions
print(pheno_df_select.shape)
print(geno_df_select.shape)


# In[20]:


### GENOTYPE BASELINE MODEL TEST ### 
# select a phenotype for genotype baseline model 
# we are going to use num_islets as the test phenotype for baseline model
phenotype = pheno_df_select["num_islets"]


# In[34]:


# ID samples with missing phenotype
samples_with_missing_pheno = phenotype[phenotype < 0].index.tolist()

print("These samples are missing phenotypes:")
print(samples_with_missing_pheno)

# select samples with known phenotypes
phenotype_complete = phenotype[~phenotype.index.isin(samples_with_missing_pheno)]

# save the complete_sample_list for identifying matching genotype
complete_sample_list = phenotype_complete.index.tolist()


# In[44]:


# select genotype
genotype_complete = geno_df_select[complete_sample_list]

print(genotype_complete.shape)

# clean snps with missing geno 
genotype_complete_dropNA = genotype_complete.dropna()

print(genotype_complete_dropNA.shape)

# import the linear model module 
import statsmodels.api as sm 


# In[ ]:


# run the linear model 
model = sm.OLS(phenotype_complete, genotype_complete_dropNA.transpose()).fit()


# In[ ]:


models.summary()

