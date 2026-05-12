import numpy as np
from rdkit import Chem


class Reagent:
    __slots__ = [
        "reagent_name",
        "smiles",
        "min_uncertainty",
        "initial_scores",
        "mol",
        "known_var",
        "current_mean",
        "current_std",
        "current_phase",
        "price",
        "num_scores",
    ]

    def __init__(self, reagent_name: str, smiles: str, price: str):
        """
        Basic init
        :param reagent_name: Reagent name
        :param smiles: smiles string
        """
        self.smiles = smiles
        self.reagent_name = reagent_name
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.initial_scores = []
        self.known_var = None  # Will be initialized during init_given_prior
        self.current_phase = "warmup"
        self.current_mean = 0
        self.current_std = 0
        self.price = price
        self.num_scores = 0

    def add_score(self, score: float):
        """
        Either adds a score to self.initial_scores if self._current_phase == "warmup", otherwise, does the bayesian
        update of the mean and standard deviation.
        :param score: New score collected for the reagent
        :return: None
        """
        if self.current_phase == "search":
            current_var = self.current_std ** 2
            mu_before = self.current_mean
            std_before = self.current_std
            print('    [add_score] reagent=%s | observed_score=%.6f' % (self.reagent_name, score))
            print('    [add_score] before: mu=%.6f, std=%.6f, current_var=%.6f, known_var=%.6f' % (
                mu_before, std_before, current_var, self.known_var))
            # Then do the bayesian update
            self.current_mean = self._update_mean(current_var=current_var, observed_value=score)
            self.current_std = self._update_std(current_var=current_var)
            self.num_scores += 1
            print('    [add_score] after:  mu=%.6f, std=%.6f (delta_mu=%+.6f, delta_std=%+.6f, num_scores=%d)' % (
                self.current_mean, self.current_std,
                self.current_mean - mu_before,
                self.current_std - std_before,
                self.num_scores))
        elif self.current_phase == "warmup":
            self.initial_scores.append(score)
            self.num_scores += 1
            print('    [add_score/warmup] reagent=%s | buffered score=%.6f (warmup count so far: %d)' % (
                self.reagent_name, score, len(self.initial_scores)))
        else:
            raise ValueError(f"self.current_phase should be warmup or search, found {self.current_phase}")
        return

    def sample(self) -> float:
        """
        Takes a random sample from the prior distribution
        :return: sample from the prior distribution
        """
        if self.current_phase != "search":
            raise ValueError(f"Must call Reagent.init() before sampling")
        return np.random.normal(loc=self.current_mean, scale=self.current_std)

    def init_given_prior(self, prior_mean: float, prior_std: float):
        """
        After warmup, set the prior distribution from the given parameters and replay the warmup scores.

        The meaning of "prior" here is the distribution before any scores have been seen for this reagent.
        This would typically be the from the score distribution seen across all reagents during the warm up phase.
        The specific values seen during warmup (stored in initial_scores) will then be run as updates just
        as they would be during the regular search phase.

        :param prior_mean: Mean of the prior distribution
        :param prior_std: Standard deviation of the prior distribution
        """
        if self.current_phase != "warmup":
            raise ValueError(f"Reagent {self.reagent_name} has already been initialized.")
        elif not self.initial_scores:
            raise ValueError(f"Must collect initial scores before initializing Reagent {self.reagent_name}")

        self.current_std = prior_std
        self.current_mean = prior_mean
        # This is an interesting assumption. Namely that the standard deviation of the
        # distribution of a reagent is estimated by the standard deviation across all reagents
        # during warmup.
        # Likely, each reagent has a smaller standard deviation than the one across all warmup
        # but this still practically works well.
        self.known_var = prior_std ** 2

        print('  [init_given_prior] reagent=%s | prior_mean=%.6f, prior_std=%.6f, known_var=%.6f | replaying %d warmup scores: %s' % (
            self.reagent_name, prior_mean, prior_std, self.known_var,
            len(self.initial_scores), self.initial_scores))

        self.current_phase = "search"

        for k, score in enumerate(self.initial_scores):
            mu_before = self.current_mean
            std_before = self.current_std
            self.add_score(score)
            print('    [init_given_prior] warmup replay %d/%d: score=%.6f | mu: %.6f -> %.6f | std: %.6f -> %.6f' % (
                k + 1, len(self.initial_scores),
                score,
                mu_before, self.current_mean,
                std_before, self.current_std))

        print('  [init_given_prior] reagent=%s | final after replay: mu=%.6f, std=%.6f' % (
            self.reagent_name, self.current_mean, self.current_std))

    def _update_mean(self, current_var: float, observed_value: float) -> float:
        """
        Bayesian update to the mean
        :param current_var: The current variance
        :param observed_value: value to use to update the mean
        :return: the updated mean
        """
        numerator = current_var * observed_value + self.known_var * self.current_mean
        denominator = current_var + self.known_var
        updated_mean = numerator / denominator
        print('    [_update_mean] (current_var * obs + known_var * mu) / (current_var + known_var)')
        print('    [_update_mean] (%.6f * %.6f + %.6f * %.6f) / (%.6f + %.6f) = %.6f / %.6f = %.6f' % (
            current_var, observed_value, self.known_var, self.current_mean,
            current_var, self.known_var,
            numerator, denominator, updated_mean))
        print('    [_update_mean] obs weight=%.4f, prior weight=%.4f (sum=1.0)' % (
            current_var / denominator, self.known_var / denominator))
        return updated_mean

    def _update_std(self, current_var: float) -> float:
        """
        Bayesian update to the standard deviation
        :param current_var: The current variance
        :return: the updated standard deviation
        """
        numerator = current_var * self.known_var
        denominator = current_var + self.known_var
        updated_std = np.sqrt(numerator / denominator)
        print('    [_update_std] sqrt(current_var * known_var / (current_var + known_var))')
        print('    [_update_std] sqrt(%.6f * %.6f / (%.6f + %.6f)) = sqrt(%.6f / %.6f) = %.6f' % (
            current_var, self.known_var, current_var, self.known_var,
            numerator, denominator, updated_std))
        return updated_std