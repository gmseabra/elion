import random
from typing import List, Optional, Tuple

import functools
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from disallow_tracker import DisallowTracker
from reagent_csv import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents_csv
from evaluators import DBEvaluator



class ThompsonSamplerCSV:
    def __init__(self, mode="maximize", db_name="SYNPLE", log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self._disallow_tracker = None
        self.hide_progress = False
        self._mode = mode
        self.db_name = db_name
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress baars
        """
        self.hide_progress = hide_progress

    def read_reagents_csv(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents_csv(reagent_file_list, self.db_name, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float]:
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        selected_reagents = []
        for idx, choice in enumerate(choice_list):
            component_reagent_list = self.reagent_lists[idx]
            selected_reagents.append(component_reagent_list[choice])
        
        prod = self.reaction.RunReactants([reagent.mol for reagent in selected_reagents])
        product_name = "_".join([reagent.reagent_name for reagent in selected_reagents])
        res = np.nan
        product_smiles = "FAIL"
        
        if prod:
            try:
                prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
                Chem.SanitizeMol(prod_mol)
                product_smiles = Chem.MolToSmiles(prod_mol)
                if isinstance(self.evaluator, DBEvaluator):
                    res = self.evaluator.evaluate(product_name)
                    res = float(res)
                else:
                    res = self.evaluator.evaluate(prod_mol)
                if np.isfinite(res):
                    print('  [evaluate] score=%.6f being added to %d reagents: %s' % (
                        res,
                        len(selected_reagents),
                        [r.reagent_name for r in selected_reagents],
                    ))
                    [reagent.add_score(res) for reagent in selected_reagents]
            except Exception as e:
                # 1. Print the actual error (e.g., AtomValenceException)
                print(f"Error during evaluation: {e}")
                
                # 2. Print the SMILES of the reagents that caused the failure
                reagent_smiles = [reagent.smiles for reagent in selected_reagents]
                print(f"Reactant SMILES causing failure: {reagent_smiles}")
                
                # 3. Fix the crash by using an f-string to handle the tuple safely
                print(f"Raw product tuple: {prod}")
                
        return product_smiles, product_name, res

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        # get the list of reagent indices
        idx_list = list(range(0, len(self.reagent_lists)))
        # get the number of reagents for each component in the reaction
        reagent_count_list = [len(x) for x in self.reagent_lists]
        warmup_results = []
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            # The number of reagents for this component
            current_max = reagent_count_list[i]
            # For each reagent...
            for j in tqdm(range(0, current_max), desc=f"Warmup {i + 1} of {len(idx_list)}", disable=self.hide_progress):
                # For each warmup trial...
                for k in range(0, num_warmup_trials):
                    current_list = [DisallowTracker.Empty] * len(idx_list)
                    current_list[i] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                    if j not in disallow_mask:
                        ## ok we can select this reagent
                        current_list[i] = j
                        # Randomly select reagents for each additional component of the reaction
                        for p in partner_list:
                            # tell the disallow tracker which site we are filling
                            current_list[p] = DisallowTracker.To_Fill
                            # get the new disallow mask
                            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                            selection_scores = np.random.uniform(size=reagent_count_list[p])
                            # null out the disallowed ones
                            selection_scores[list(disallow_mask)] = np.nan
                            # and select a random one
                            current_list[p] = np.nanargmax(selection_scores).item(0)
                        self._disallow_tracker.update(current_list)
                        product_smiles, product_name, score = self.evaluate(current_list)
                        if np.isfinite(score):
                            warmup_results.append([score, product_smiles, product_name])

        warmup_scores = [ws[0] for ws in warmup_results]
        self.logger.info(
            f"warmup score stats: "
            f"cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores):0.4f}, "
            f"std={np.std(warmup_scores):0.4f}, "
            f"min={np.min(warmup_scores):0.4f}, "
            f"max={np.max(warmup_scores):0.4f}")
        # initialize each reagent
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        print('[warm_up] prior_mean=%.6f | prior_std=%.6f | num_warmup_scores=%d' % (prior_mean, prior_std, len(warmup_scores)))
        for i in range(0, len(self.reagent_lists)):
            for j in range(0, len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                    print('  [warm_up] component %d, reagent %d (%s): init mu=%.6f, std=%.6f, num_scores=%d' % (
                        i, j, reagent.reagent_name,
                        reagent.current_mean, reagent.current_std, reagent.num_scores,
                    ))
                except ValueError:
                    self.logger.info(f"Skipping reagent {reagent.reagent_name} because there were no successful evaluations during warmup")
                    self._disallow_tracker.retire_one_synthon(i, j)
        self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        return warmup_results

    def search(self, num_cycles=25):
        """Run the search
        :param: num_cycles: number of search iterations
        :return: a list of SMILES and scores
        """
        out_list = []
        rng = np.random.default_rng()
        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
            for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
                reagent_list = self.reagent_lists[cycle_id]
                selected_reagents[cycle_id] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                stds = np.array([r.current_std for r in reagent_list])
                mu = np.array([r.current_mean for r in reagent_list])
                score_counts = np.array([r.num_scores for r in reagent_list])
                print('--- cycle_id (component being selected): %d ---' % cycle_id)
                print('stds: %s' % stds)
                print('mu: %s' % mu)
                print('score_counts (how many times each reagent has been evaluated): %s' % score_counts)
                print('mu range: [%.6f, %.6f], std range: [%.6f, %.6f]' % (mu.min(), mu.max(), stds.min(), stds.max()))
                choice_row = rng.normal(size=len(reagent_list)) * stds + mu
                print('choice_row: %s' % choice_row)
                if disallow_mask:
                    choice_row[np.array(list(disallow_mask))] = np.nan
                    print('disallow_mask (retired/excluded reagent indices): %s' % disallow_mask)
                winner_idx = self.pick_function(choice_row)
                print('winner_idx: %d | winner mu: %.6f | winner std: %.6f | winner score_count: %d' % (
                    winner_idx,
                    reagent_list[winner_idx].current_mean,
                    reagent_list[winner_idx].current_std,
                    reagent_list[winner_idx].num_scores,
                ))
                selected_reagents[cycle_id] = winner_idx
                print('selected_reagents: %s' % selected_reagents)
            self._disallow_tracker.update(selected_reagents)
            print('self._disallow_tracker: %s' % self._disallow_tracker)
            # Select a reagent for each component, according to the choice function
            smiles, name, score = self.evaluate(selected_reagents)
            print('=== iteration %d result ===' % i)
            print('score: %s | smiles: %s | name: %s' % (score, smiles, name))
            if np.isfinite(score):
                # Print posterior belief AFTER update (add_score was called inside evaluate)
                for comp_idx, reagent_idx in enumerate(selected_reagents):
                    r = self.reagent_lists[comp_idx][reagent_idx]
                    print('  post-update | component %d, reagent %d (%s): mu=%.6f, std=%.6f, num_scores=%d' % (
                        comp_idx, reagent_idx, r.reagent_name,
                        r.current_mean, r.current_std, r.num_scores,
                    ))
            print('')
            if np.isfinite(score):
                out_list.append([score, smiles, name])
            if i % 100 == 0:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
        return out_list