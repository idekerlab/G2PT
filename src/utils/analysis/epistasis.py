from src.utils.tree import SNPTreeParser
import pandas as pd
from sgkit.io import plink
from scipy.stats import chi2_contingency
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from scipy.stats import chi2

class EpistasisFinder(object):
    def __init__(self, tree_parser : SNPTreeParser, bfile, attention_results, cov=None):
        self.tree_parser = tree_parser
        plink_data = plink.read_plink(path=bfile)
        self.genotype = pd.DataFrame(plink_data.call_genotype.as_numpy().sum(axis=-1).T)
        self.genotype.index = plink_data.sample_id.values
        self.genotype.columns = plink_data.variant_id.values
        print("From PLINK %d variants with %d samples are queried" % (self.genotype.shape[1], self.genotype.shape[0]))
        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        if cov is not None:
            self.cov_df = pd.read_csv(cov, sep='\t')
        else:
            self.cov_df = pd.DataFrame({'FID': plink_data.sample_family_id.as_numpy(),
                                        'IID': plink_data.sample_id.as_numpy(),
                                        'SEX': plink_data.sample_sex.as_numpy(),
                                        'PHENOTYPE': plink_data.sample_phenotype.as_numpy() })
            self.cov_df = self.cov_df[['FID', 'IID', 'SEX', 'PHENOTYPE']]
            self.cov_df = self.cov_df.loc[self.cov_df.PHENOTYPE!=-1]
            self.cov_df['PHENOTYPE'] = self.cov_df['PHENOTYPE'] - 1
            self.genotype = self.genotype.loc[self.cov_df.IID]
        self.cov_df['FID'] = self.cov_df['FID'].astype(str)
        self.cov_df['IID'] = self.cov_df['IID'].astype(str)
        if type(attention_results) == str:
            self.attention_results = pd.read_csv(attention_results)
        elif type(attention_results) == pd.DataFrame:
            self.attention_results = attention_results

        self.attention_results_0 = self.attention_results.loc[self.attention_results.SEX == 0]
        self.attention_results_1 = self.attention_results.loc[self.attention_results.SEX == 1]

    def search_epistasis_on_system(self, system, sex=0, quantile=0.9, verbose=0):
        target_snps = self.tree_parser.sys2snp[system]
        n_target_snps = len(target_snps)
        if sex == 0:
            attention_results = self.attention_results_0
        else:
            attention_results = self.attention_results_1
        thr = attention_results[system].quantile(quantile)
        print("System: %s, Sex: %d"%(system, sex))
        print("Running Chi-Square Test...")
        if quantile >=0.5:
            risky_samples = attention_results.loc[attention_results[system] >= thr].IID.map(str)
        else:
            risky_samples = attention_results.loc[attention_results[system] <= thr].IID.map(str)
        genotype = self.genotype.loc[attention_results.IID.map(str)]
        result_chi = []
        print("\tTesting %d SNPs on %d risky individuals"%(len(target_snps), len(risky_samples)))
        for target_snp in target_snps:
            df, p_val, _, _ = self.get_snp_chi_sqaure(genotype, risky_samples, target_snp)
            if p_val * n_target_snps < 0.05:
                result_chi.append(target_snp)
                if verbose==1:
                    print(f'\t\t{target_snp} passes Chi-Square test with p-value {p_val}')
        print(f'\tFrom {n_target_snps} SNPs, {len(result_chi)} SNPs pass Chi-Square test')
        print('Running Fisher')
        sig_snp_df = genotype.loc[risky_samples, result_chi]
        odd_result, p_df = self.calculate_fisher(sig_snp_df)
        sig_snp_pairs = self.get_significant_pairs_from_fisher(p_df, verbose=verbose)
        n = sig_snp_df.shape[1]
        print(f'\t From {(n*(n-1)/2)} significant pairs, {len(sig_snp_pairs)} pairs pass Fisher test')
        return sig_snp_pairs

    def get_snp_chi_sqaure(self, snp_df, risky_samples, target_snp):
        with_snp_risk = snp_df.loc[risky_samples, target_snp].sum()
        without_snp_risk = snp_df.loc[risky_samples].shape[0] * 2 - with_snp_risk
        with_snp_population = snp_df.loc[:, target_snp].sum()
        without_snp_population = snp_df.shape[0] * 2 - with_snp_population
        data = {'Risky Subset': [with_snp_risk, without_snp_risk],
                'Population': [with_snp_population, without_snp_population]}
        chr, loc, alt, ref = target_snp.split(':')
        result_df = pd.DataFrame(data, index=[alt, ref])
        # print(df.T)
        chi2, p_value, dof, expected = chi2_contingency(result_df.T)
        return result_df.T, p_value, chi2, dof

    def calculate_fisher(self, filtered_snp_df):
        odds_ratios = pd.DataFrame(np.zeros((filtered_snp_df.shape[1], filtered_snp_df.shape[1])), index=filtered_snp_df.columns, columns=filtered_snp_df.columns)
        p_values = pd.DataFrame(np.zeros((filtered_snp_df.shape[1], filtered_snp_df.shape[1])), index=filtered_snp_df.columns, columns=filtered_snp_df.columns)

        for i in range(filtered_snp_df.shape[1]):
            for j in range(filtered_snp_df.shape[1]):
                if i == j:
                    odds_ratios.iloc[i, j] = 1  # df.shape[0]
                    p_values.iloc[i, j] = 1
                else:
                    try:
                        contingency_table = pd.crosstab(filtered_snp_df.iloc[:, i], filtered_snp_df.iloc[:, j])
                        odds_ratio, p_value, dof, expected = chi2_contingency(contingency_table)
                    except:
                        odds_ratio, p_value = np.inf, 1
                    odds_ratios.iloc[
                        i, j] = odds_ratio  # ((df.iloc[:, i] == df.iloc[:, j]) & (df.iloc[:, i] != 0)).sum()/df.shape[0]
                    odds_ratios.iloc[j, i] = odds_ratios.iloc[i, j]  # Symmetric matrix
                    p_values.iloc[
                        i, j] = p_value  # ((df.iloc[:, i] == df.iloc[:, j]) & (df.iloc[:, i] != 0)).sum()/df.shape[0]
                    p_values.iloc[j, i] = p_values.iloc[i, j]  # Symmetric matrix
        return odds_ratios, p_values

    def get_significant_pairs_from_fisher(self, fisher_df_results, verbose=0):
        queried_epistasis = []

        n = fisher_df_results.shape[0]
        mask = fisher_df_results < (0.05 / ((n * (n - 1))/2))
        row_indices, col_indices = np.where(mask)
        # Convert indices to row-column tuples
        tuples = [(fisher_df_results.index[row], fisher_df_results.columns[col]) for row, col in zip(row_indices, col_indices) if
                  self.tree_parser.snp2gene[fisher_df_results.index[row]] != self.tree_parser.snp2gene[fisher_df_results.columns[col]]]
        tuples = set(tuple(sorted(t)) for t in tuples)
        for snp_1, snp_2 in tuples:
            if self.tree_parser.snp2gene[snp_1] == self.tree_parser.snp2gene[snp_2]:
                continue
            # queried_epistasis[key][(row, col)] = df.loc[row, col]#+= tuples
            if verbose == 1:
                print(f"\t\tEpistatic interaction between {snp_1} -> {self.tree_parser.snp2gene[snp_1]} and {snp_2} -> {self.tree_parser.snp2gene[snp_2]} is detected ")
            queried_epistasis.append((snp_1, snp_2))#[key][(row, col)] = df.loc[row, col]  # += tuples
        return queried_epistasis

    def calculate_epistatic_significance(self, snp_df, pairs):
        sig_snps = list(set(element for tup in pairs for element in tup))
        partial_genotype =
        significant_epistasis = []
        for snp_1, snp_2 in pairs:


    def rename_snp(self, snp):
        new_name =  "_".join(reversed(snp.split(":")))
        return new_name

    def rollback_snp_name(self, new_name):
        orig_name = ":".join(reversed(new_name.split("_")))
        return orig_name


