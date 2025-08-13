from src.utils.tree import SNPTreeParser
import pandas as pd
from sgkit.io import plink
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from scipy.stats import chi2
import itertools
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


class EpistasisFinder(object):
    """
    Finds and analyzes epistatic interactions between SNPs within biological systems.

    This class integrates attention scores from a trained model with genotype data
    to perform a multi-stage statistical analysis. It first identifies candidate
    SNPs based on their relevance to a system (using attention scores), then
    filters SNP pairs by physical distance, tests for pairwise interaction using
    Fisher's Exact Test, and finally validates significant pairs using a
    regression model to confirm the statistical interaction effect.

    Attributes:
        tree_parser (SNPTreeParser): An instance of the parser containing SNP,
            gene, and system relationships.
        genotype (pd.DataFrame): A DataFrame of genotype data, with samples as
            rows and SNPs as columns.
        cov_df (pd.DataFrame): A DataFrame containing covariate and phenotype data.
        attention_results (pd.DataFrame): A DataFrame of attention scores for each
            sample and system.
    """
    def __init__(self, tree_parser : SNPTreeParser, attention_results, tsv_path, cov, pheno=None, flip=False):
        """
        Initializes the EpistasisFinder and loads all necessary data from TSV files.

        Args:
            tree_parser (SNPTreeParser): An initialized SNPTreeParser object.
            attention_results (str or pd.DataFrame): Path to a CSV file or a DataFrame of attention scores.
            tsv_path (str): Path to the directory containing genotypes.tsv and snp2gene.tsv.
            cov (str): Path to a tab-separated covariate file.
            pheno (str, optional): Path to a tab-separated phenotype file. Defaults to None.
            flip (bool, optional): If True, swaps reference and alternate alleles. Defaults to False.
        """
        self.tree_parser = tree_parser
        self.flip = flip

        # Load genotype data from TSV
        genotype_file = f"{tsv_path}/genotypes.tsv"
        self.genotype = pd.read_csv(genotype_file, sep='	', index_col=0)
        self.genotype.index = self.genotype.index.astype(str)
        self.genotype.columns = self.genotype.columns.astype(str)
        if flip:
            self.genotype = 2 - self.genotype
            print("Swapping Ref and Alt!")
        
        # Load SNP metadata from snp2gene file
        snp2gene_file = f"{tsv_path}/snp2gene.tsv"
        snp2gene_df = pd.read_csv(snp2gene_file, sep='\t')
        self.snp2chr = dict(zip(snp2gene_df['snp'], snp2gene_df['chr']))
        self.snp2pos = dict(zip(snp2gene_df['snp'], snp2gene_df['pos']))
        print(f"From TSV {self.genotype.shape[1]} variants with {self.genotype.shape[0]} samples are queried")

        self.cov_df = pd.read_csv(cov, sep='\t')
        self.cov_df['FID'] = self.cov_df['FID'].astype(str)
        self.cov_df['IID'] = self.cov_df['IID'].astype(str)

        if pheno is not None:
            pheno_df = pd.read_csv(pheno, sep='\t')
            pheno_df['IID'] = pheno_df['IID'].astype(str)
            pheno_df['FID'] = pheno_df['FID'].astype(str)
            if 'PHENOTYPE' in self.cov_df.columns:
                self.cov_df = self.cov_df.drop(columns=['PHENOTYPE'])
            self.cov_df = pd.merge(self.cov_df, pheno_df, on=['FID', 'IID'])

        # Align dataframes
        self.cov_df = self.cov_df.loc[self.cov_df['IID'].isin(self.genotype.index)]
        self.genotype = self.genotype.loc[self.cov_df.IID]

        self.cov_ids = [c for c in self.cov_df.columns if c not in ['FID', 'IID', 'PHENOTYPE']]
        
        if isinstance(attention_results, str):
            self.attention_results = pd.read_csv(attention_results)
        else:
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

    def search_epistasis_on_system(self, system, sex=0, quantile=0.9, chi=True, fisher=True, return_significant_only=True, check_inheritance=True, verbose=0,
                                   snp_inheritance_dict = {}, binary=False, target='PHENOTYPE'):
        """
        Searches for epistatic interactions for a given biological system.

        This method executes a multi-step pipeline:
        1.  Determines the optimal inheritance model for each SNP (optional).
        2.  Filters SNPs using a Chi-Square test based on attention scores to
            identify those prevalent in a high-risk cohort.
        3.  Generates all pairs of the filtered SNPs.
        4.  Filters out SNP pairs that are physically close on the chromosome.
        5.  Performs Fisher's Exact Test on the distant pairs to find
            statistically significant co-occurrences (optional).
        6.  Uses a regression model to test for a statistical interaction
            effect for the remaining pairs, correcting for multiple testing.

        Args:
            system (str): The name of the system (e.g., GO term) to analyze.
            sex (int, optional): The sex to include in the analysis (0, 1, or 2
                for all). Defaults to 0.
            quantile (float, optional): The attention score quantile to define the
                high-risk cohort. Defaults to 0.9.
            chi (bool, optional): Whether to perform the Chi-square Test
                step. Defaults to True.
            fisher (bool, optional): Whether to perform the Fisher's Exact Test
                step. Defaults to True.
            return_significant_only (bool, optional): If True, returns only the
                pairs that are statistically significant after all tests. If
                False, returns results for all tested pairs. Defaults to True.
            check_inheritance (bool, optional): If True, determines the best-fit
                inheritance model for each SNP before testing. Defaults to True.
            verbose (int, optional): Verbosity level (0 or 1). Defaults to 0.
            snp_inheritance_dict (dict, optional): A pre-computed dictionary of
                SNP inheritance models. Defaults to an empty dict.
            binary (bool, optional): Whether the target phenotype is binary (for
                logistic regression) or continuous (for linear regression).
                Defaults to False.
            target (str, optional): The name of the phenotype column in the
                covariate data. Defaults to 'PHENOTYPE'.

        Returns:
            tuple: A tuple containing:
                - list: A list of tuples for significant epistatic pairs. Each
                  tuple contains (snp1, snp2, raw_p_value, fdr_adjusted_p_value).
                - dict: An updated dictionary of determined SNP inheritance models.
        """
        target_snps = self.tree_parser.sys2snp[system]
        n_target_snps = len(target_snps)
        if snp_inheritance_dict is None:
            snp_inheritance_dict = {}

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
                    genotype_for_ineritance_model = genotype_merged_with_covs[[str(target_snp), target]+self.cov_ids]
                    if sex!=2:
                        genotype_for_ineritance_model = genotype_for_ineritance_model.loc[genotype_for_ineritance_model.SEX==sex]
                    target_snp_type = self.determine_inheritance_model(genotype_for_ineritance_model[[str(target_snp)]+self.cov_ids], genotype_for_ineritance_model['PHENOTYPE'].values, target_snp, verbose=verbose)
                    snp_inheritance_dict[target_snp] = target_snp_type


        if quantile >=0.5:
            risky_samples = attention_results.loc[attention_results[system] >= thr].IID.map(str)
        else:
            risky_samples = attention_results.loc[attention_results[system] <= thr].IID.map(str)

        if chi:
            print("Running Chi-Square Test...")
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

            # Generate all possible pairs from Chi-square results
            all_snp_pairs = list(itertools.combinations(result_chi, 2))
        else:
            all_snp_pairs = list(itertools.combinations(target_snps, 2))

        # 1. Filter out pairs that are too close physically
        print('Filtering Close SNPs...')
        distant_snp_pairs = [pair for pair in all_snp_pairs if self.check_distance(pair[0], pair[1])]
        print(f"\tFrom {len(all_snp_pairs)} pairs, {len(all_snp_pairs) - len(distant_snp_pairs)} proximal pairs were removed, leaving {len(distant_snp_pairs)} for testing.")

        sig_snp_pairs = distant_snp_pairs
        if fisher and distant_snp_pairs:
            # 2. Perform Fisher's Exact Test on the remaining distant pairs
            print("Running Fisher's Exact Test on distant pairs...")
            sig_snp_pairs = self._run_fisher_on_pairs(risky_samples, distant_snp_pairs, snp_inheritance_dict, verbose=verbose)
            print(f'\tFrom {len(distant_snp_pairs)} pairs, {len(sig_snp_pairs)} passed Fisher test with FDR correction.')
        
        if not sig_snp_pairs:
             return [], snp_inheritance_dict

        print('Calculating statistical Interaction p-value ')
        final_significant_pairs = self.get_statistical_epistatic_significance(sig_snp_pairs, attention_results.IID.map(str),
                                                                    snp_inheritance_dict=snp_inheritance_dict,
                                                                    verbose=verbose,
                                                                    return_significant_only=return_significant_only,
                                                                    target=target, binary=binary)
        print(f'\tFrom {len(sig_snp_pairs)} pairs, {len(final_significant_pairs)} significant interactions were found after regression analysis.')
        
        return final_significant_pairs, snp_inheritance_dict

    def _run_fisher_on_pairs(self, risky_samples, snp_pairs, snp_inheritance_dict, verbose=0):
        """
        Performs Fisher's exact test on a pre-filtered list of SNP pairs.

        This private helper function iterates through a list of SNP pairs,
        calculates the contingency table for each pair within the `risky_samples`
        cohort, and performs Fisher's exact test. It then applies a Benjamini-Hochberg
        FDR correction to the resulting p-values.

        Args:
            risky_samples (list): A list of sample IDs defining the high-risk cohort.
            snp_pairs (list): A list of tuples, where each tuple is a pair of SNP IDs.
            snp_inheritance_dict (dict): A dictionary mapping SNPs to their
                inheritance models ('additive', 'dominant', etc.).
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            list: A list of SNP pair tuples that were statistically significant
                  after FDR correction.
        """
        if not snp_pairs:
            return []

        target_snps = list(map(str, set(itertools.chain.from_iterable(snp_pairs))))
        partial_genotype = self.genotype.loc[risky_samples, target_snps].copy()
        for snp in target_snps:
            if snp in snp_inheritance_dict:
                partial_genotype = self.model_encoders[snp_inheritance_dict[snp]](partial_genotype, snp)

        raw_pvals = []
        for snp1, snp2 in snp_pairs:
            contingency_table = pd.crosstab(partial_genotype[str(snp1)], partial_genotype[str(snp2)])
            if contingency_table.shape == (2, 2):
                _, p_value = fisher_exact(contingency_table)
                raw_pvals.append(p_value)
            else:
                raw_pvals.append(1.0)

        if not raw_pvals:
            return []

        reject_flags, fdr_corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')

        significant_pairs = []
        for i, (snp1, snp2) in enumerate(snp_pairs):
            if reject_flags[i]:
                if verbose == 1:
                    print(f"\t\tInteraction between {snp1} and {snp2} is significant after FDR (p={fdr_corrected_pvals[i]:.4g})")
                significant_pairs.append((snp1, snp2))
        
        return significant_pairs

    def get_snp_chi_sqaure(self, population, risky_samples, target_snp, snp_inheritance_dict={}):
        """
        Performs a Chi-Square test for a single SNP.

        This test determines if the allele frequency of a given SNP is
        significantly different between the `risky_samples` cohort and the
        total `population`.

        Args:
            population (list): List of all sample IDs.
            risky_samples (list): List of sample IDs in the high-risk cohort.
            target_snp (str): The SNP identifier to test.
            snp_inheritance_dict (dict, optional): Maps SNPs to inheritance models.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The contingency table used for the test.
                - float: The p-value from the test.
                - float: The Chi-Square statistic.
                - int: The degrees of freedom.
        """
        partial_genotype = self.genotype.loc[population, [str(target_snp)]].copy()
        if target_snp in snp_inheritance_dict.keys():
            partial_genotype = self.model_encoders[snp_inheritance_dict[target_snp]](partial_genotype, target_snp)

        with_snp_risk = partial_genotype.loc[risky_samples, str(target_snp)].sum()
        without_snp_risk = partial_genotype.loc[risky_samples].shape[0] * 2 - with_snp_risk
        with_snp_population = partial_genotype.loc[:, str(target_snp)].sum()
        without_snp_population = partial_genotype.shape[0] * 2 - with_snp_population
        data = {'Risky Subset': [with_snp_risk, without_snp_risk],
                'Population': [with_snp_population, without_snp_population]}
        #chr, loc, alt, ref = target_snp.split(':')
        #changed code
        result_df = pd.DataFrame(data, index=['REF', "ALT"])
        # print(df.T)
        try:
            chi2, p_value, dof, expected = chi2_contingency(result_df.T)
        except ValueError:
            p_value = 1.0
            chi2 = 0
            dof = 0
            expected = None
        return result_df.T, p_value, chi2, dof

    def get_statistical_epistatic_significance(self, pairs, cohort,
                                               snps_in_system=(),
                                               return_significant_only=True,
                                               snp_inheritance_dict={}, target='PHENOTYPE',
                                               binary=False,
                                               verbose=0):
        """
        Tests for a statistical interaction effect between SNP pairs using regression.

        For each pair, this method fits a regression model with an interaction
        term (e.g., `phenotype ~ snp1 + snp2 + snp1:snp2 + covariates`). It
        then uses the p-value of the interaction term to determine if a
        significant epistatic effect exists. P-values are corrected for multiple
        testing using the Benjamini-Hochberg FDR method.

        Args:
            pairs (list): A list of SNP pair tuples to test.
            cohort (list): A list of all sample IDs to include in the analysis.
            snps_in_system (list, optional): A list of all SNPs in the system,
                used for creating the initial genotype DataFrame. Defaults to ().
            return_significant_only (bool, optional): If True, returns only
                significant pairs. Defaults to True.
            snp_inheritance_dict (dict, optional): Maps SNPs to inheritance models.
            target (str, optional): The name of the phenotype column.
            binary (bool, optional): If True, use logistic regression. If False,
                use ordinary least squares. Defaults to False.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            list: A list of tuples for epistatic pairs. If `return_significant_only`
                  is True, only significant pairs are returned. Each tuple contains
                  (snp1, snp2, raw_p_value, fdr_adjusted_p_value).
        """

        target_snps = list(set(element for tup in pairs for element in tup))
        #print(pairs, target_snps)
        #print(self.genotype.columns)
        if len(snps_in_system) == 0:
            partial_genotype = self.genotype.loc[cohort, [str(snp) for snp in target_snps]].copy()
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
            formula_no_epistasis = target+' ~ ' + ' + '.join(self.cov_ids) + " + %s + %s"%(snp_1_renamed, snp_2_renamed)
            combination_name = "%s:%s"%(snp_1_renamed, snp_2_renamed)
            formula_epistasis = formula_no_epistasis + " + " + combination_name
            #md_reduced = smf.ols(formula_no_epistasis, data=partial_genotype_cov_merged).fit()
            try:
                if binary:
                    md_full = smf.logit(formula_epistasis, data=partial_genotype_cov_merged).fit()
                else:
                    md_full = smf.ols(formula_epistasis, data=partial_genotype_cov_merged).fit()
                #ll_full = md_full.llf
                #ll_reduced = md_reduced.llf
                #lr_stat = 2 * (ll_full - ll_reduced)
                #df_diff = md_full.df_model - md_reduced.df_model
                #combinatory_pvalue = stats.chi2.sf(lr_stat, df_diff)
                combinatory_pvalue = md_full.pvalues[combination_name]
                raw_results.append((snp_1, snp_2, combinatory_pvalue))
            except:
                raw_results.append((snp_1, snp_2, 1))

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
        """Formats a SNP ID to be compatible with regression formula syntax."""
        new_name = "SNP" + "_".join(reversed(str(snp).split(":")))
        return new_name

    def rollback_snp_name(self, new_name):
        """Reverts a formatted SNP ID back to its original format."""
        orig_name = ":".join(reversed(new_name.split("_")))
        return orig_name

    def check_distance(self, snp_1, snp_2, distance_threshold=500000):
        """
        Checks if two SNPs are on different chromosomes or far apart on the same one.

        Args:
            snp_1 (str): The ID of the first SNP.
            snp_2 (str): The ID of the second SNP.
            distance_threshold (int, optional): The minimum distance (in base
                pairs) to be considered "distant". Defaults to 500000.

        Returns:
            bool: True if the SNPs are distant, False otherwise.
        """
        if self.snp2chr[snp_1] != self.snp2chr[snp_2]:
            return True
        else:
            if np.abs(self.snp2pos[snp_1]-self.snp2pos[snp_2]) > distance_threshold:
                return True
            else:
                return False

    @staticmethod
    def code_additivity(genotype, snp_id):
        """Encodes genotype for an additive inheritance model (no change)."""
        return genotype

    @staticmethod
    def code_dominance(genotype, snp_id):
        """Encodes genotype for a dominant inheritance model (AA=0, Aa=1, aa=1)."""
        genotype.loc[:, str(snp_id)] = genotype[str(snp_id)].replace(2, 1)
        return genotype

    @staticmethod
    def code_recessive(genotype, snp_id):
        """Encodes genotype for a recessive inheritance model (AA=0, Aa=0, aa=1)."""
        genotype.loc[:, str(snp_id)] = genotype[str(snp_id)].replace({1: 0, 2: 1})#.replace(1, 0)
        return genotype

    @staticmethod
    def code_overdominance(genotype, snp_id):
        """Encodes genotype for an overdominant model (AA=0, Aa=1, aa=0)."""
        genotype.loc[:, str(snp_id)] = genotype[str(snp_id)].replace(2, 0)
        return genotype

    @staticmethod
    def code_underdominance(genotype, snp_id):
        """Encodes genotype for an underdominant model (AA=1, Aa=0, aa=1)."""
        genotype.loc[:, str(snp_id)] = genotype[str(snp_id)].replace({1: 0, 0: 1, 2: 1})
        return genotype

    def determine_inheritance_model(self, genotypes, phenotype, target_snp, verbose=0, loss='aic'):
        """
        Determines the best-fit inheritance model for a SNP using AIC or BIC.

        This method fits five different regression models (additive, dominant,
        recessive, overdominant, underdominant) for a single SNP against a
        phenotype and selects the model with the lowest Akaike Information
        Criterion (AIC) or Bayesian Information Criterion (BIC).

        Args:
            genotypes (pd.DataFrame): A DataFrame containing genotype and
                covariate data for samples.
            phenotype (array-like): An array of phenotype values.
            target_snp (str): The SNP ID to evaluate.
            verbose (int, optional): Verbosity level. Defaults to 0.
            loss (str, optional): The criterion to use for model selection,
                either 'aic' or 'bic'. Defaults to 'aic'.

        Returns:
            str: The name of the best-fit inheritance model.
        """
        # Define the model coding functions in a dict

        model_aics = {}
        for model_name, encoder in self.model_encoders.items():
            coded_g = encoder(genotypes.copy(), target_snp)
            if coded_g.shape[0] == 0:
                if verbose > 0:
                    print(f"Skipping {target_snp} for model {model_name} due to no valid genotypes.")
                model_aics[model_name] = float('inf')
                continue
            
            try:
                ols_model = sm.OLS(phenotype, coded_g.values).fit()
                if loss == 'aic':
                    model_aics[model_name] = ols_model.aic
                else:
                    model_aics[model_name] = ols_model.bic
            except ValueError:
                if verbose > 0:
                    print(f"Could not fit model for {target_snp} with {model_name} model.")
                model_aics[model_name] = float('inf')


        if not model_aics or all(v == float('inf') for v in model_aics.values()):
             return 'additive' # Return default if no model could be fit

        best_model = min(model_aics, key=model_aics.get)

        if verbose == 1:
            print(f'{target_snp} is {best_model}, AIC: {model_aics}')
        return best_model

    def merge_cov_df(self, new_cov_df, left_on=None, right_on=None):
        """Merges a new covariate DataFrame with the existing one."""
        self.cov_df = self.cov_df.merge(new_cov_df, left_on=left_on, right_on=right_on)

    def draw_epistasis(self, target_snp_0, target_snp_1, phenotype, sex=None, figsize=(22, 5), estimator='mean',
                           errorbar='ci', out_dir=None, regression=False):
        """
        Generates and displays interaction plots for a pair of SNPs.

        This function creates a series of point plots to visualize how the
        genotypes of two SNPs interact to affect a given phenotype.

        Args:
            target_snp_0 (str): The ID of the first SNP.
            target_snp_1 (str): The ID of the second SNP.
            phenotype (str): The name of the phenotype column to plot.
            sex (int, optional): The sex to include (0, 1, or None for all).
                Defaults to None.
            figsize (tuple, optional): The figure size. Defaults to (22, 5).
            estimator (str, optional): The statistical estimator to use (e.g.,
                'mean', 'median'). Defaults to 'mean'.
            errorbar (str, optional): The error bar style (e.g., 'ci', 'sd').
                Defaults to 'ci'.
            out_dir (str, optional): If provided, the path to save the figure.
                Defaults to None.
            regression (bool, optional): This argument is present but not
                currently used in the function. Defaults to False.
        """
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
        if errorbar is None:
            errorbar='se'

        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_0, ax=axes[0], estimator=estimator, errorbar=errorbar,
                      order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],)
        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_1, ax=axes[1], estimator=estimator, errorbar=errorbar,
                      order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],)

        sns.pointpoint(data=cov_df_partial, y=phenotype, x=target_snp_0, hue=target_snp_1,
                      hue_order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],
                      order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],
                      ax=axes[2], estimator=estimator, errorbar=errorbar)
        sns.pointplot(data=cov_df_partial, y=phenotype, x=target_snp_1, hue=target_snp_0,
                      order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],
                      hue_order=['Homozygous ref.', 'Heterozygous', 'Homozygous alt.'],
                      ax=axes[3], estimator=estimator, errorbar=errorbar)
        if out_dir is not None:
            plt.savefig(out_dir)
        plt.show()