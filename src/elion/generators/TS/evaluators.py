import importlib
import os
import warnings
from abc import ABC, abstractmethod

# Genuinely module-level: every evaluator needs these.
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ── Per-class OPTIONAL dependencies ─────────────────────────────────────────
# Every import below is used by exactly ONE evaluator class. Importing them at
# module level made a package that your configured evaluator never calls fatal
# to `import generators.TS` -> Generator.__init__ -> the whole elion.py run,
# before a single line of TS code executed. And they failed ONE AT A TIME, so
# each install exposed the next: useful_rdkit_utils, then sqlitedict, then...
#
# openeye already had this treatment (it is commercial, so it was obviously
# optional). The others are optional for the same reason and just weren't
# recognised as such.
#
# Missing packages are collected and reported in ONE warning at import, then
# raised individually — with the pip line — only if you actually select the
# class that needs one.
_MISSING: "dict[str, str]" = {}


def _optional(module: str, pypi: str, used_by: str):
    """Import `module`, or record it as missing and return None."""
    try:
        return importlib.import_module(module)
    except ImportError:                                   # pragma: no cover
        _MISSING[module] = f"{pypi:<28} (needed by {used_by})"
        return None


# NOTE useful_rdkit_utils has a name mismatch: PyPI hyphenates, import underscores.
uru        = _optional("useful_rdkit_utils", "useful-rdkit-utils>=0.2.7",
                       "MWEvaluator, MLClassifierEvaluator")
joblib     = _optional("joblib", "joblib", "MLClassifierEvaluator")
pd         = _optional("pandas", "pandas>=2.0", "LookupEvaluator")
_sqlitedic = _optional("sqlitedict", "sqlitedict", "DBEvaluator")
SqliteDict = getattr(_sqlitedic, "SqliteDict", None)

if _MISSING:
    warnings.warn(
        "TS evaluators: %d optional package(s) not installed in this "
        "environment. The evaluators that need them are disabled; every other "
        "evaluator — including ElionEstimatorEvaluator, which is what "
        "input_TS.yml selects — is unaffected.\n%s\n"
        "Install all of them at once with:\n    pip install %s"
        % (len(_MISSING),
           "\n".join("    " + v for v in _MISSING.values()),
           " ".join(f'"{v.split("(")[0].strip()}"' for v in _MISSING.values()))
    )


def _require(obj, module: str, pypi: str, cls_name: str) -> None:
    """Fail at USE time with an actionable message, not at import time."""
    if obj is None:
        raise ImportError(
            f"{cls_name} requires the optional package {module!r}, which is not "
            f"installed in this environment. Install it with:\n"
            f'    pip install "{pypi}"'
        )


try:
    from openeye import oechem
    from openeye import oeomega
    from openeye import oeshape
    from openeye import oedocking
except ImportError:
    # Since openeye is a commercial software package, just pass with a warning if not available
    warnings.warn(f"Openeye packages not available in this environment; do not attempt to use ROCSEvaluator or "
                  f"FredEvaluator")

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, mol):
        pass

    @property
    @abstractmethod
    def counter(self):
        pass


class MWEvaluator(Evaluator):
    """A simple evaluation class that calculates molecular weight, this was just a development tool
    """

    def __init__(self):
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        _require(uru, 'useful_rdkit_utils', 'useful-rdkit-utils>=0.2.7', 'MWEvaluator')
        self.num_evaluations += 1
        return uru.MolWt(mol)


class FPEvaluator(Evaluator):
    """An evaluator class that calculates a fingerprint Tanimoto to a reference molecule
    """

    def __init__(self, input_dict):
        self.ref_smiles = input_dict["query_smiles"]
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()
        self.ref_mol = Chem.MolFromSmiles(self.ref_smiles)
        self.ref_fp = self.fpgen.GetFingerprint(self.ref_mol)
        self.num_evaluations = 0
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, rd_mol_in):
        self.num_evaluations += 1
        rd_mol_fp = self.fpgen.GetFingerprint(rd_mol_in)
        return DataStructs.TanimotoSimilarity(self.ref_fp, rd_mol_fp)


class ROCSEvaluator(Evaluator):
    """An evaluator class that calculates a ROCS score to a reference molecule
    """

    def __init__(self, input_dict):
        ref_filename = input_dict['query_molfile']
        ref_fs = oechem.oemolistream(ref_filename)
        self.ref_mol = oechem.OEMol()
        oechem.OEReadMolecule(ref_fs, self.ref_mol)
        self.max_confs = 50
        self.score_cache = {}
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, rd_mol_in):
        """Generate conformers with Omega and evaluate the ROCS overlay of conformers to a reference molecule
        :param rd_mol_in: Input RDKit molecule
        :return: ROCS Tanimoto Combo score, returns -1 if conformer generation fails
        """
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(rd_mol_in)
        # Look up to see if we already processed this molecule
        arc_tc = self.score_cache.get(smi)
        if arc_tc is not None:
            tc = arc_tc
        else:
            fit_mol = oechem.OEMol()
            oechem.OEParseSmiles(fit_mol, smi)
            ret_code = generate_confs(fit_mol, self.max_confs)
            if ret_code:
                tc = self.overlay(fit_mol)
            else:
                tc = -1.0
            self.score_cache[smi] = tc
        return tc

    def overlay(self, fit_mol):
        """Use ROCS to overlay two molecules
        :param fit_mol: OEMolecule
        :return: Combo Tanimoto for the overlay
        """
        prep = oeshape.OEOverlapPrep()
        prep.Prep(self.ref_mol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(self.ref_mol)
        prep.Prep(fit_mol)
        score = oeshape.OEBestOverlayScore()
        overlay.BestOverlay(score, fit_mol, oeshape.OEHighestTanimoto())
        return score.GetTanimotoCombo()


class LookupEvaluator(Evaluator):
    """A simple evaluation class that looks up values from a file.
    This is primarily used for testing.
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        ref_filename = input_dictionary['ref_filename']
        ref_colname = input_dictionary['ref_colname']
        _require(pd, 'pandas', 'pandas>=2.0', 'LookupEvaluator')   # both branches below need it
        if ref_filename.endswith(".csv"):
            ref_df = pd.read_csv(ref_filename)
        elif ref_filename.endswith(".parquet"):
            ref_df = pd.read_parquet(ref_filename)
        else:
            print(ref_filename,"does not have valid extendsion must be in [.csv,.parquet]")
            assert(False)
        self.ref_dict = dict([(a, b) for a, b in ref_df[['SMILES', ref_colname]].values])

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        val = self.ref_dict.get(smi)
        if val is not None:
            return val
        else:
            return np.nan

class DBEvaluator(Evaluator):
    """A simple evaluator class that looks up values from a database.
    This is primarily used for benchmarking
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        self.db_prefix = input_dictionary['db_prefix']
        db_filename = input_dictionary['db_filename']
        _require(SqliteDict, 'sqlitedict', 'sqlitedict', 'DBEvaluator')
        self.ref_dict = SqliteDict(db_filename)

    def __repr__(self):
        return "DBEvalutor"


    @property
    def counter(self):
        return self.num_evaluations


    def evaluate(self, smiles):
        self.num_evaluations += 1
        res = self.ref_dict.get(f"{self.db_prefix}{smiles}")
        if res is None:
            return np.nan
        else:
            if res == -500:
                return np.nan
            return res
    

class FredEvaluator(Evaluator):
    """An evaluator class that docks a molecule with the OEDocking Toolkit and returns the score
    """

    def __init__(self, input_dict):
        du_file = input_dict["design_unit_file"]
        if not os.path.isfile(du_file):
            raise FileNotFoundError(f"{du_file} was not found or is a directory")
        self.dock = read_design_unit(du_file)
        self.num_evaluations = 0
        self.max_confs = 50

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        mc_mol = oechem.OEMol()
        oechem.OEParseSmiles(mc_mol, smi)
        confs_ok = generate_confs(mc_mol, self.max_confs)
        score = 1000.0
        docked_mol = oechem.OEGraphMol()
        if confs_ok:
            ret_code = self.dock.DockMultiConformerMolecule(docked_mol, mc_mol)
        else:
            ret_code = oedocking.OEDockingReturnCode_ConformerGenError
        if ret_code == oedocking.OEDockingReturnCode_Success:
            dock_opts = oedocking.OEDockOptions()
            sd_tag = oedocking.OEDockMethodGetName(dock_opts.GetScoreMethod())
            # this is a stupid hack, I need to figure out how to do this correctly
            oedocking.OESetSDScore(docked_mol, self.dock, sd_tag)
            score = float(oechem.OEGetSDData(docked_mol, sd_tag))
        return score


def generate_confs(mol, max_confs):
    """Generate conformers with Omega
    :param max_confs: maximum number of conformers to generate
    :param mol: input OEMolecule
    :return: Boolean Omega return code indicating success of conformer generation
    """
    rms = 0.5
    strict_stereo = False
    omega = oeomega.OEOmega()
    omega.SetRMSThreshold(rms)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetStrictStereo(strict_stereo)
    omega.SetMaxConfs(max_confs)
    error_level = oechem.OEThrow.GetLevel()
    # Turn off OEChem warnings
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    status = omega(mol)
    # Turn OEChem warnings back on
    oechem.OEThrow.SetLevel(error_level)
    return status


def read_design_unit(filename):
    """Read an OpenEye design unit
    :param filename: design unit filename (.oedu)
    :return: a docking grid
    """
    du = oechem.OEDesignUnit()
    rfs = oechem.oeifstream()
    if not rfs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    du = oechem.OEDesignUnit()
    if not oechem.OEReadDesignUnit(rfs, du):
        oechem.OEThrow.Fatal("Failed to read design unit")
    if not du.HasReceptor():
        oechem.OEThrow.Fatal("Design unit %s does not contain a receptor" % du.GetTitle())
    dock_opts = oedocking.OEDockOptions()
    dock = oedocking.OEDock(dock_opts)
    dock.Initialize(du)
    return dock


def test_fred_eval():
    """Test function for the Fred docking Evaluator
    :return: None
    """
    input_dict = {"design_unit_file": "data/2zdt_receptor.oedu"}
    fred_eval = FredEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = fred_eval.evaluate(mol)
    print(score)


def test_rocs_eval():
    """Test function for the ROCS evaluator
    :return: None
    """
    input_dict = {"query_molfile": "data/2chw_lig.sdf"}
    rocs_eval = ROCSEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    combo_score = rocs_eval.evaluate(mol)
    print(combo_score)


class MLClassifierEvaluator(Evaluator):
    """An evaluator class the calculates a score based on a trained ML model
    """

    def __init__(self, input_dict):
        _require(joblib, 'joblib', 'joblib', 'MLClassifierEvaluator')
        self.cls = joblib.load(input_dict["model_filename"])
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        _require(uru, 'useful_rdkit_utils', 'useful-rdkit-utils>=0.2.7', 'MLClassifierEvaluator')
        self.num_evaluations += 1
        fp = uru.mol2morgan_fp(mol)
        return self.cls.predict_proba([fp])[:,1][0]


def test_ml_classifier_eval():
    """Test function for the ML Classifier Evaluator
    :return: None
    """
    input_dict = {"model_filename": "mapk1_modl.pkl"}
    ml_cls_eval = MLClassifierEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = ml_cls_eval.evaluate(mol)
    print(score)


class ElionEstimatorEvaluator(Evaluator):
    """Wraps Elion's Estimators to conform to the Evaluator interface.
    Evaluates one molecule at a time via estimate_properties -> estimate_rewards -> TOTAL.
    """

    def __init__(self, properties_cfg: dict):
        """
        :param properties_cfg: The 'Reward_function' sub-dict from the Elion YAML config.
        """
        from properties.Estimators import Estimators
        self.estimator = Estimators(properties_cfg)
        self.num_evaluations = 0

    @property
    def counter(self) -> int:
        return self.num_evaluations

    def evaluate(self, mol) -> float:
        """
        :param mol: RDKit ROMol
        :return: total reward as a float, or np.nan on failure

        Single-molecule path. Kept for the Evaluator interface, but the TS
        search should never take it — see evaluate_batch() below.
        """
        predictions = self.estimator.estimate_properties([mol])
        rewards = self.estimator.estimate_rewards(predictions)
        self.num_evaluations += 1
        return float(rewards["TOTAL"][0])

    def evaluate_batch(self, mols) -> list:
        """Score a whole batch in ONE estimator call. Order-preserving.

        WHY THIS EXISTS
        ---------------
        `thompson_sampling.evaluate_batch()` already does:

            if hasattr(self.evaluator, 'evaluate_batch'):
                scores = self.evaluator.evaluate_batch(valid_mols)
            else:
                scores = [self.evaluator.evaluate(m) for m in valid_mols]

        and its comment names *this class* as the intended fast path — but the
        method was never written, so `hasattr` was always False and every run
        took the per-molecule fallback. With `CHEMBERT_BE` at rew_coeff 0.95
        that means `CHEMBERT_BE.predict()` builds a `SMILES_Dataset` and spins
        up a **DataLoader per molecule**, then runs a one-row forward pass.
        The symptom is the status line reading

            properties.Estimators: _cls: <...QED_Score...> | n=1 predictions

        once per molecule, and ~5 s/it at eval_batch_size 256.

        `Estimators` was always batch-ready: `estimate_properties` takes a list
        and stashes `__n_mols__` in the returned dict precisely so batch calls
        of differing size cannot race on instance state. Only this method was
        missing.

        :param mols: sequence of RDKit ROMol
        :return: list of total rewards, one per input, same order
        """
        _mols = list(mols)
        if not _mols:
            return []

        try:
            predictions = self.estimator.estimate_properties(_mols)
            rewards     = self.estimator.estimate_rewards(predictions)
            totals      = list(rewards["TOTAL"])
            # Estimators.estimate_rewards already hard-fails on a short property
            # list, but the TOTAL length is what the caller zips against
            # valid_indices — a silent mismatch would misattribute scores to the
            # wrong reagents, which is far worse than an exception.
            if len(totals) != len(_mols):
                raise ValueError(
                    f"estimator returned {len(totals)} TOTAL scores for "
                    f"{len(_mols)} molecules")
            self.num_evaluations += len(_mols)
            return [float(t) for t in totals]

        except Exception as exc:
            # Rescue the run rather than lose it — but say so LOUDLY and once.
            # A silent fallback here reads exactly like the bug this method was
            # written to fix: correct results, ~50x slower, no explanation.
            if not getattr(self, "_batch_fallback_warned", False):
                self._batch_fallback_warned = True
                warnings.warn(
                    f"ElionEstimatorEvaluator.evaluate_batch failed on a batch of "
                    f"{len(_mols)} ({type(exc).__name__}: {exc}). Falling back to "
                    f"per-molecule scoring for the REST OF THIS RUN — expect a "
                    f"large slowdown. If this is a GPU OOM, lower "
                    f"generator.TS.eval_batch_size in input_TS.yml.")
            out = []
            for m in _mols:
                try:
                    out.append(self.evaluate(m))
                except Exception:
                    out.append(float(np.nan))
            return out

    def calculate_properties(config):
        """Given a SMILES file, calculate the properties of the molecules.	"""	
        smiles_file = Path(config['Control']['smiles_file'])
        output_file = Path(config['Control']['output_smi_file'])
        estimator = Estimators(config['Reward_function'])

        mols, smis = utils.read_smi_file(smiles_file)
        predictions = estimator.estimate_properties(mols)

        if config['Control']['verbosity'] > 0:
            utils.print_results(smis, predictions, header="PROPERTIES")
        else:
            utils.print_stats(predictions, header="STATISTICS", print_header=True)
            
        utils.save_smi_file(output_file, smis, predictions)

def test_Elion_classifier_eval():
    """Test function for the ML Classifier Evaluator
    :return: None
    """
    import sys
    # Add the directory containing 'input_reader.py'
    sys.path.append('../..')
    import input_reader
    from pathlib import Path
    config = input_reader.read_input_file('../../input_TS.yml')
    #-- Calculation Type --#
    run_type = config['Control']['run_type']
    elion_cls_eval = ElionEstimatorEvaluator(config['Reward_function'])
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = elion_cls_eval.evaluate(mol)
    print(score)

if __name__ == "__main__":
    test_Elion_classifier_eval()