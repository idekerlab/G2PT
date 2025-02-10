from src.utils.tree import SNPTreeParser
import pandas as pd
from sgkit.io import plink
from scipy.stats import chi2_contingency
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


class EpistasisFinder(object):
    """
    A class to identify and analyze epistatic interactions in genetic data using attention results, and statistical methods.
    """
    def __init__(self, tree_parser : SNPTreeParser, bfile, attention_results, cov=None, flip=False):
        """
        Initialize the EpistasisFinder class.

        Args:
            tree_parser (SNPTreeParser): An object for parsing SNP and tree-related data.
            bfile (str): Path to the PLINK binary file containing genotype data.
            attention_results (str or pd.DataFrame): Path to a file or DataFrame containing attention-based results.
            cov (str, optional): Path to a file containing covariate data. If not provided, default covariates are used.

        Attributes:
            tree_parser (SNPTreeParser): Reference to the provided tree parser.
            genotype (pd.DataFrame): Genotype data extracted from the PLINK file.
            cov_df (pd.DataFrame): Covariate data.
            attention_results (pd.DataFrame): Attention results for SNPs.
            attention_results_0 (pd.DataFrame): Attention results for sex 0.
            attention_results_1 (pd.DataFrame): Attention results for sex 1.
        """
        self.tree_parser = tree_parser
        self.plink_data = plink.read_plink(path=bfile)
        self.genotype = pd.DataFrame(self.plink_data.call_genotype.as_numpy().sum(axis=-1).T)
        self.flip = flip
        if flip:
            self.genotype = 2 - self.genotype
            print("Swapping Ref and Alt!")
        self.genotype.index = self.plink_data.sample_id.values
        self.genotype.columns = self.plink_data.variant_id.values
        self.snp2chr = {snp:self.plink_data.contig_id.values[chrome] for snp, chrome in zip(self.plink_data.variant_id.values, self.plink_data.variant_contig.values)}
        self.snp2pos = {snp:pos for snp, pos in zip(self.plink_data.variant_id.values, self.plink_data.variant_position.values)}
        print("From PLINK %d variants with %d samples are queried" % (self.genotype.shape[1], self.genotype.shape[0]))
        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        if cov is not None:
            self.cov_df = pd.read_csv(cov, sep='\t')
        else:
            self.cov_df = pd.DataFrame({'FID': self.plink_data.sample_family_id.as_numpy(),
                                        'IID': self.plink_data.sample_id.as_numpy(),
                                        'SEX': self.plink_data.sample_sex.as_numpy(),
                                        'PHENOTYPE': self.plink_data.sample_phenotype.as_numpy() })
            self.cov_df = self.cov_df[['FID', 'IID', 'SEX', 'PHENOTYPE']]
            self.cov_df = self.cov_df.loc[self.cov_df.PHENOTYPE!=-1]
            self.cov_df['PHENOTYPE'] = self.cov_df['PHENOTYPE'] - 1
            self.genotype = self.genotype.loc[self.cov_df.IID]
        self.cov_df['FID'] = self.cov_df['FID'].astype(str)
        self.cov_df['IID'] = self.cov_df['IID'].astype(str)
        self.cov_ids = [cov for cov in self.cov_df.columns[2:] if cov != 'PHENOTYPE']

        if type(attention_results) == str:
            self.attention_results = pd.read_csv(attention_results)
        elif type(attention_results) == pd.DataFrame:
            self.attention_results = attention_results

        self.attention_results_0 = self.attention_results.loc[self.attention_results.SEX == 0]
        self.attention_results_1 = self.attention_results.loc[self.attention_results.SEX == 1]

        self.model_encoders = {
            'additive': self.code_additivity,
            'dominant': self.code_dominance,
            'recessive': self.code_recessive,
            'overdominant': self.code_overdominance,
            'underdominant': self.code_underdominance
        }

    def search_epistasis_on_system(self, system, sex=0, quantile=0.9, return_significant_only=True, check_inheritance=True, verbose=0,
                                   snp_inheritance_dict = {}):
        """
        Search for epistatic interactions in a specified biological system.

        Args:
            system (str): The biological system under analysis.
            sex (int, optional): The sex for filtering data (0 or 1). Default is 0.
            quantile (float, optional): Quantile threshold for attention results. Default is 0.9.
            verbose (int, optional): Verbosity level. Default is 0.

        Returns:
            list: List of significant SNP pairs with epistatic interactions.
        """
        target_snps = self.tree_parser.sys2snp[system]
        n_target_snps = len(target_snps)
        if snp_inheritance_dict is None:
            snp_inheritance_dict = {}

        '''
        n_dominant = len(dominant_snps)
        n_recessive = len(recessive_snps)
        n_overdominant = len(recessive_snps)
        n_underdominant = len(underdominant_snps)
        n_typical = n_target_snps - n_dominant - n_recessive - n_overdominant - n_underdominant

        print(f'From {n_target_snps}, there are {n_typical} typical SNPs, {n_dominant} dominant SNPs, {n_recessive} recessive SNPs, {n_overdominant} overdominant SNPs, {n_underdominant} underdominant SNPs,')
        '''
        if sex == 0:
            attention_results = self.attention_results_0
        elif sex == 1:
            attention_results = self.attention_results_1
        else:
            attention_results =  self.attention_results
        thr = attention_results[system].quantile(quantile)


        print("System: %s, Sex: %d"%(system, sex))

        if check_inheritance:
            print("Check Inheritance")
            genotype_merged_with_covs = self.genotype.merge(self.cov_df, left_index=True, right_on='IID')
            for target_snp in target_snps:
                if target_snp not in snp_inheritance_dict.keys():
                    genotype_for_ineritance_model = genotype_merged_with_covs[[target_snp, 'PHENOTYPE']+self.cov_ids]
                    if sex!=2:
                        genotype_for_ineritance_model = genotype_for_ineritance_model.loc[genotype_for_ineritance_model.SEX==sex]
                    target_snp_type = self.determine_inheritance_model(genotype_for_ineritance_model[[target_snp]+self.cov_ids], genotype_for_ineritance_model['PHENOTYPE'].values, target_snp, verbose=verbose)
                    snp_inheritance_dict[target_snp] = target_snp_type

        print("Running Chi-Square Test...")



        if quantile >=0.5:
            risky_samples = attention_results.loc[attention_results[system] >= thr].IID.map(str)
        else:
            risky_samples = attention_results.loc[attention_results[system] <= thr].IID.map(str)

        result_chi = []
        print("\tTesting %d SNPs on %d risky individuals"%(len(target_snps), len(risky_samples)))
        for target_snp in target_snps:
            df, p_val, _, _ = self.get_snp_chi_sqaure(attention_results.IID.map(str), risky_samples, target_snp,
                                                      snp_inheritance_dict=snp_inheritance_dict)
            if p_val * n_target_snps < 0.05:
                result_chi.append(target_snp)
                if verbose==1:
                    print(f'\t\t{target_snp} -> {self.tree_parser.snp2gene[target_snp]} passes Chi-Square test with p-value {p_val}')
        print(f'\tFrom {n_target_snps} SNPs, {len(result_chi)} SNPs pass Chi-Square test')
        print('Running Fisher')

        odd_result, p_df = self.calculate_fisher(risky_samples, result_chi, snp_inheritance_dict=snp_inheritance_dict)
        sig_snp_pairs = self.get_significant_pairs_from_fisher(p_df, verbose=verbose)
        n = odd_result.shape[1]
        print(f'\tFrom {(n*(n-1)/2)} significant pairs, {len(sig_snp_pairs)} pairs pass Fisher test')
        print('Filtering Close SNPs')
        distant_snp_pairs = [(snp_1, snp_2) for snp_1, snp_2 in sig_snp_pairs if self.check_distance(snp_1, snp_2)]
        print(f"\tFrom {len(sig_snp_pairs)} pairs, close {len(sig_snp_pairs)-len(distant_snp_pairs)} SNP pairs are removed")

        print('Calculating statistical Interaction p-value ')



        sig_snp_pairs = self.get_statistical_epistatic_significance(distant_snp_pairs, attention_results.IID.map(str),
                                                                    snp_inheritance_dict=snp_inheritance_dict,
                                                                    verbose=verbose,
                                                                    return_significant_only=return_significant_only)
        print(f'\tFrom {len(distant_snp_pairs)} pairs, {len(sig_snp_pairs)} significant interaction are queried')
        '''
        if len(sig_snp_pairs)==0:
            if quantile >=0.5:
                non_risky_samples = attention_results.loc[attention_results[system] <= 1-thr].IID.map(str)
            else:
                non_risky_samples = attention_results.loc[attention_results[system] >= 1-thr].IID.map(str)
            samples = risky_samples.tolist() + non_risky_samples.tolist()
            print("No Epistasis Found, now compare top and bottom")
            sig_snp_pairs = self.get_statistical_epistatic_significance(distant_snp_pairs, samples, verbose=verbose)
            print(f'\tFrom {len(distant_snp_pairs)} pairs, {len(sig_snp_pairs)} significant interaction are queried')
        '''
        return sig_snp_pairs, snp_inheritance_dict

    def get_snp_chi_sqaure(self, population, risky_samples, target_snp, snp_inheritance_dict={}):
        """
        Perform a chi-square test for a given SNP.

        Args:
            genotype (pd.DataFrame): DataFrame of SNP genotypes.
            risky_samples (list): List of samples with high-risk phenotypes.
            target_snp (str): SNP identifier.

        Returns:
            tuple: Chi-square result DataFrame, p-value, chi-square statistic, and degrees of freedom.
        """
        partial_genotype = self.genotype.loc[population, [target_snp]].copy()
        if target_snp in snp_inheritance_dict.keys():
            partial_genotype = self.model_encoders[snp_inheritance_dict[target_snp]](partial_genotype, target_snp)

        with_snp_risk = partial_genotype.loc[risky_samples, target_snp].sum()
        without_snp_risk = partial_genotype.loc[risky_samples].shape[0] * 2 - with_snp_risk
        with_snp_population = partial_genotype.loc[:, target_snp].sum()
        without_snp_population = partial_genotype.shape[0] * 2 - with_snp_population
        data = {'Risky Subset': [with_snp_risk, without_snp_risk],
                'Population': [with_snp_population, without_snp_population]}
        chr, loc, alt, ref = target_snp.split(':')
        result_df = pd.DataFrame(data, index=[alt, ref])
        # print(df.T)
        chi2, p_value, dof, expected = chi2_contingency(result_df.T)
        return result_df.T, p_value, chi2, dof

    def calculate_fisher(self, risky_samples, target_snps, snp_inheritance_dict={}):
        partial_genotype = self.genotype.loc[risky_samples, target_snps].copy()
        for target_snp in target_snps:
            if target_snp in snp_inheritance_dict.keys():
                partial_genotype = self.model_encoders[snp_inheritance_dict[target_snp]](partial_genotype, target_snp)

        """
        Perform Fisher's exact test on pairs of SNPs.

        Args:
            filtered_snp_df (pd.DataFrame): DataFrame of filtered SNPs.

        Returns:
            tuple: Odds ratios and p-values DataFrames.
        """
        odds_ratios = pd.DataFrame(np.zeros((partial_genotype.shape[1], partial_genotype.shape[1])), index=partial_genotype.columns, columns=partial_genotype.columns)
        p_values = pd.DataFrame(np.zeros((partial_genotype.shape[1], partial_genotype.shape[1])), index=partial_genotype.columns, columns=partial_genotype.columns)

        for i in range(partial_genotype.shape[1]):
            for j in range(partial_genotype.shape[1]):
                if i == j:
                    odds_ratios.iloc[i, j] = 1  # df.shape[0]
                    p_values.iloc[i, j] = 1
                else:
                    try:
                        contingency_table = pd.crosstab(partial_genotype.iloc[:, i], partial_genotype.iloc[:, j])
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
        """
        Identify significant SNP pairs from Fisher's exact test results.

        Args:
            fisher_df_results (pd.DataFrame): DataFrame of Fisher's exact test p-values.
            verbose (int, optional): Verbosity level. Default is 0.

        Returns:
            list: List of significant SNP pairs.
        """
        queried_epistasis = []
        n = fisher_df_results.shape[0]
        if n <= 1:
            return []
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
                print(f"\t\tEpistatic interaction between {snp_1} -> {self.tree_parser.snp2gene[snp_1]} and {snp_2} -> {self.tree_parser.snp2gene[snp_2]} is detected, p-value: {fisher_df_results.loc[snp_1, snp_2]} ")
            queried_epistasis.append((snp_1, snp_2))#[key][(row, col)] = df.loc[row, col]  # += tuples
        return queried_epistasis

    def get_statistical_epistatic_significance(self, pairs, cohort,
                                               snps_in_system=(),
                                               return_significant_only=True,
                                               snp_inheritance_dict={}, verbose=0):
        """
        Evaluate statistical significance of epistatic interactions using regression models.

        Args:
            pairs (list): List of SNP pairs.
            cohort (list): List of sample IDs in the cohort.
            verbose (int, optional): Verbosity level. Default is 0.

        Returns:
            list: List of statistically significant SNP pairs with epistatic interactions.
        """

        target_snps = list(set(element for tup in pairs for element in tup))
        if len(snps_in_system) == 0:
            partial_genotype = self.genotype.loc[cohort, target_snps].copy()
        else:
            partial_genotype = self.genotype.loc[cohort, snps_in_system].copy()
        for target_snp in target_snps:
            if target_snp in snp_inheritance_dict.keys():
                partial_genotype = self.model_encoders[snp_inheritance_dict[target_snp]](partial_genotype, target_snp)


        partial_genotype.columns = map(self.rename_snp, partial_genotype.columns.tolist())
        partial_genotype_cov_merged = partial_genotype.merge(self.cov_df, left_index=True, right_on='IID')
        print(f"\tTesting {len(target_snps)} SNPs on {partial_genotype.shape[0]} individuals, SNPs in system {snps_in_system}")

        raw_results = []
        significant_epistasis = []


        for snp_1, snp_2 in pairs:
            if len(snps_in_system) == 0:
                snps_in_system = [snp_1, snp_2]
            snps_in_system_renamed = [self.rename_snp(snp) for snp in snps_in_system]
            snp_1_renamed = self.rename_snp(snp_1)
            snp_2_renamed = self.rename_snp(snp_2)
            formula_no_epistasis = 'PHENOTYPE ~ ' + ' + '.join(self.cov_ids) + " + %s + %s"%(snp_1_renamed, snp_2_renamed)
            combination_name = "%s:%s"%(snp_1_renamed, snp_2_renamed)
            formula_epistasis = formula_no_epistasis + " + " + combination_name
            #md_reduced = smf.ols(formula_no_epistasis, data=partial_genotype_cov_merged).fit()
            md_full = smf.ols(formula_epistasis, data=partial_genotype_cov_merged).fit()
            #ll_full = md_full.llf
            #ll_reduced = md_reduced.llf
            #lr_stat = 2 * (ll_full - ll_reduced)
            #df_diff = md_full.df_model - md_reduced.df_model
            #combinatory_pvalue = stats.chi2.sf(lr_stat, df_diff)
            combinatory_pvalue = md_full.pvalues[combination_name]
            raw_results.append((snp_1, snp_2, combinatory_pvalue))

        alpha = 0.05
        if len(pairs) == 0:
            return []
        raw_pvals = [r[2] for r in raw_results]
        reject_flags, fdr_corrected_pvals, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')

        # 3) Construct final list including the adjusted p-values
        significant_epistasis = []
        all_epistasis = []

        for i, (snp_1, snp_2, raw_p) in enumerate(raw_results):
            adj_p = fdr_corrected_pvals[i]
            is_significant = reject_flags[i]

            # Save the result in a unified tuple
            result_tuple = (snp_1, snp_2, raw_p, adj_p)

            if is_significant:
                # If significant, optionally print details
                if verbose == 1:
                    print(f'Epistatic interaction between {snp_1} -> {self.tree_parser.snp2gene[snp_1]} '
                          f'and {snp_2} -> {self.tree_parser.snp2gene[snp_2]} is SIGNIFICANT '
                          f'(raw_p={raw_p:.5g}, FDR_adj_p={adj_p:.5g})')
                significant_epistasis.append(result_tuple)
            else:
                if verbose == 1:
                    print(f'Epistatic interaction between {snp_1} -> {self.tree_parser.snp2gene[snp_1]} '
                          f'and {snp_2} -> {self.tree_parser.snp2gene[snp_2]} is NOT SIGNIFICANT '
                          f'(raw_p={raw_p:.5g}, FDR_adj_p={adj_p:.5g})')

            # Keep all results, if the user wants them
            all_epistasis.append(result_tuple)

        if return_significant_only:
            return significant_epistasis
        else:
            return all_epistasis

    def rename_snp(self, snp):
        new_name =  "_".join(reversed(snp.split(":")))
        return new_name

    def rollback_snp_name(self, new_name):
        orig_name = ":".join(reversed(new_name.split("_")))
        return orig_name

    def check_distance(self, snp_1, snp_2, distance_threshold=500000):
        if self.snp2chr[snp_1] != self.snp2chr[snp_2]:
            return True
        else:
            if np.abs(self.snp2pos[snp_1]-self.snp2pos[snp_2]) > distance_threshold:
                return True
            else:
                return False

    @staticmethod
    def code_additivity(genotype, snp_id):
        return genotype

    @staticmethod
    def code_dominance(genotype, snp_id):
        genotype.loc[:, snp_id] = genotype[snp_id].replace(2, 1)
        return genotype

    @staticmethod
    def code_recessive(genotype, snp_id):
        genotype.loc[:, snp_id] = genotype[snp_id].replace({1: 0, 2: 1})#.replace(1, 0)
        return genotype

    @staticmethod
    def code_overdominance(genotype, snp_id):
        genotype.loc[:, snp_id] = genotype[snp_id].replace(2, 0)
        return genotype

    @staticmethod
    def code_underdominance(genotype, snp_id):
        genotype.loc[:, snp_id] = genotype[snp_id].replace({1: 0, 0: 1, 2: 1})
        return genotype

    def determine_inheritance_model(self, genotypes, phenotype, target_snp, verbose=0):
        """
        Determine which inheritance model (among typical/additive, dominant, recessive,
        overdominant, underdominant) best fits a continuous phenotype using AIC.

        Parameters
        ----------
        genotypes : array-like
            Genotypes coded as 0, 1, or 2 for each individual.
        phenotype : array-like
            Continuous phenotype values for each individual, same length as genotypes.

        Returns
        -------
        best_model : str
            One of {'typical', 'dominant', 'recessive', 'overdominant', 'underdominant'}
        model_aics : dict
            A mapping of model name -> AIC value, for inspection.
        """
        # Define the model coding functions in a dict

        model_aics = {}
        for model_name, encoder in self.model_encoders.items():
            coded_g = encoder(genotypes, target_snp)
            ols_model = sm.OLS(phenotype, coded_g.values).fit()
            model_aics[model_name] = ols_model.aic
        best_model = min(model_aics, key=model_aics.get)
        if verbose == 1:
            print(f'{target_snp} is {best_model}, AIC: {model_aics}')
        return best_model

    def merge_cov_df(self, new_cov_df, left_on=None, right_on=None):
        self.cov_df = self.cov_df.merge(new_cov_df, left_on=left_on, right_on=right_on)

    def draw_epistasis(self, target_snp_0, target_snp_1, phenotype, sex=None, figsize=(22, 5), out_dir=None):
        genotype_partial = self.genotype[[target_snp_0, target_snp_1]]
        target_snps_0_a0_index = genotype_partial.loc[(genotype_partial[target_snp_0] == 0)].index
        target_snps_0_hetero_index = genotype_partial[(genotype_partial[target_snp_0] == 1)].index
        target_snps_0_a1_index = genotype_partial.loc[(genotype_partial[target_snp_0] == 2)].index

        target_snps_1_a0_index = genotype_partial.loc[(genotype_partial[target_snp_1] == 0)].index
        target_snps_1_hetero_index = genotype_partial.loc[(genotype_partial[target_snp_1] == 1)].index
        target_snps_1_a1_index = genotype_partial.loc[(genotype_partial[target_snp_1] == 2)].index

        cov_df_partial = self.cov_df[['IID', 'SEX', phenotype]]
        cov_df_partial = cov_df_partial.set_index('IID')

        cov_df_partial.loc[target_snps_0_a0_index, target_snp_0] = 'Homozygous ref.'
        cov_df_partial.loc[target_snps_0_hetero_index, target_snp_0] = 'Heterozygous'
        cov_df_partial.loc[target_snps_0_a1_index, target_snp_0] = 'Homozygous alt.'

        cov_df_partial.loc[target_snps_1_a0_index, target_snp_1] = 'Homozygous ref.'
        cov_df_partial.loc[target_snps_1_hetero_index, target_snp_1] = 'Heterozygous'
        cov_df_partial.loc[target_snps_1_a1_index, target_snp_1] = 'Homozygous alt.'

        fig, axes = plt.subplots(1, 4, figsize=figsize)
        axes = axes.ravel()
        if sex is None:
            cov_df_partial = cov_df_partial
        elif sex == 0:
            cov_df_partial = cov_df_partial.loc[cov_df_partial.SEX == 0]
        elif sex == 1:
            cov_df_partial = cov_df_partial.loc[cov_df_partial.SEX == 1]

        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_0, ax=axes[0])
        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_1, ax=axes[1])

        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_0, hue=target_snp_1,
                      hue_order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'], ax=axes[2])
        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_1, hue=target_snp_0,
                      hue_order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'], ax=axes[3])
        if out_dir is not None:
            plt.savefig(out_dir)
        plt.show()


