
import sys
from types import new_class
import numpy as np
from pydoc import doc
from typing import Tuple, List, Dict
from multiprocessing import Pool

# Chemistry
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import QED
from rdkit.Chem.rdmolops import RDKFingerprint

# Local 
from .SA_Score.sascore import SA_Scorer
sascorer = SA_Scorer()

"""
Functions to calculate properties and rewards. 

Every "property" function receives a rdkit.Chem.Mol object, and returns 
a float corresponding to that property, and a dictionary of other settings
for that property calculation.

Every "reward" function recceives a float (the property value), and a 
dictionary of other relevant settings for the reward calculation.

(Usually, there's only one dictionary per property with the definitions
relevant for both property and reward calculation.)

Every "reward" function has _the same name_ as the property function, with 
an "_reward" added to end of the name. For example, for 'prob_active' 
property, 

    Function to calculate PROPERTY: prob_active(Mol, **kwargs)
    Function to calculate REWARD  : prob_active_reward(score, **kwargs)

This convention is important because
the function names are generated dynamically as needed. 
"""

# Define a random generator to be used
rng = np.random.default_rng()

# =============================================================================
#           BATCH CALCULATORS
# =============================================================================
# Tests Scores and Rewards
#
# One can first call "estimate_properties" to calculate all properties for a batch
# of molecules, then call "estimate rewards" with those properties to calculate the 
# corresponding rewards.

def estimate_properties_parallel(molecules:List[str], properties_to_estimate:Dict, n_jobs:int=16) -> Dict:
    """Estimates all properties for a batch of molecules

    Args:
        molecules (List[str]): List of SMILES strings to calculate rewards
        properties_to_estimate (Dict): The dictionary with the definitions and
                                       parameters for the properties to be estimated

    Returns:
        Dict: Dictionary where each 'key' is a property name, and the 'values' are
              lists of the respective property values.
    """


    # Create the dictionary for properties    
    properties = {}

    mols = [Chem.MolFromSmiles(mol, sanitize=True) for mol in molecules]

    for prop_name, params in properties_to_estimate.items():

        if prop_name == 'prob_active' and params['model_type'] == "CHEMBERT":
            from properties.activity.CHEMBERT.chembert import SMILES_Dataset
            dataset = SMILES_Dataset(molecules)
            properties[prop_name] = np.array(params['predictor'].predict(dataset))
            #print("In development...")
            #quit()
        else:
            # Dynamically create the function name to be called
            prop = getattr(sys.modules[__name__], prop_name)
            with Pool(processes=n_jobs) as p:
                properties[prop_name] = p.starmap(auxiliary_prop_calc,[(prop, mol, params) for mol in mols ] )

    return properties

def auxiliary_prop_calc(prop, mol, kwargs):
    return prop(mol, **kwargs)

def estimate_rewards_batch(properties:Dict,reward_params:Dict) -> Dict:
    """Estimates all the rewards for a batch of molecules
       Here, we expect the properties have already been calculated, 
       for example by calling `estimate_properties_batch`, then stored
       in the `properties` dict passed in.

    Args:
        properties (Dict): Dictionary with pre-calculated properties
        reward_params (Dict): Dictionary with the definitions and parameters
                              for each property

    Returns:
        Dict: Dictionary where the 'keys' are the property names, and the 
              'values' are lists with the respective property values.
    """

    # Create the dictionary for rewards    
    rewards = {}
    for prop_name in properties.keys():
        #rewards[prop_name] = []

        # Dynamically generate the function name to call. The convention
        # here is that the name is ALWAYS propname_reward
        prop = getattr(sys.modules[__name__], f"{prop_name}_reward")

        prop_rewards = []
        for prop_value in properties[prop_name]:
            this_reward = ( prop(prop_value, **reward_params[prop_name]) )
            prop_rewards.append(this_reward)
        rewards[prop_name] = np.array(prop_rewards)
    
    # Adds a column for "Total" rewards. 
    rewards["TOTAL"] = estimate_total_reward_batch(rewards,reward_params)
    return rewards

def estimate_total_reward_batch(rewards_dict:Dict,reward_params:Dict) -> List[float]:
    """ Calculates the sum of all rewards for each molecule. This is just the
        simple sum of all rewards, as long as the prob_active reward is > 1
        (prob_active is >= threshold).

    Args:
        rewards_dict (Dict): The pre-calculated rewards
        reward_params (Dict): Definitions and parameters for each property

    Returns:
        List[float]: Sum of rewards.
    """
    # There's probably a better way to do this, with numpy!

    # Determine the lenght of the list
    n_samples = len( rewards_dict[ list(rewards_dict.keys())[0] ]  )

    # Create the list for the total rewards    
    rewards = []

    for molecule in range(n_samples):

        # It only makes sense to consider other scores if the 
        # prob_active is adequate, so let's start with it.
        this_reward = rewards_dict['prob_active'][molecule]

        # The lowest reward for prob_active is 1. We only consider molecules
        # with scores larger than that.
        if this_reward > 1:
            for prop_name in rewards_dict.keys():
                if prop_name not in ['prob_active','TOTAL']:
                    this_reward = this_reward + rewards_dict[prop_name][molecule]         
        rewards.append(this_reward)
    return rewards

def estimate_capped_rewards_batch(properties: Dict, reward_params: Dict) -> Dict:
    """ Attemps to estimate the rewards in a "capped" way, where the total
        reward will be a simple step function (1|15). In this case, each 
        property will need a 'threshold', which may or may not be adjustable.

        Here, we expect the properties have already been calculated, 
        for example by calling estimate_properties_batch, and passed in
        in the `properties` dict.

        
    Args:
        properties (Dict): Dictionary with the pre-calculated properties
        reward_params (Dict): Dictionary with the definitinos and parameters for
                              each property.

    Returns:
        Dict: Dictionary where the 'keys' are the property names, and the 'values'
              are the respective property rewards for all molecules in the batch.
    """

    # Determines the lenght of the lists
    n_mols = len( properties['prob_active'] )

    # Initialize the rewards dict
    rewards = {}
    probabilities = []
    props = list(properties.keys())

    # Mandatory properties
    props.remove('prob_active')
    props.remove('scaffold_match') 

    # This will get the coeff of all the other properties and
    # transform into probabilities. The sum of all probabilities
    # must be 1.
    for prop in props:
        probabilities.append(reward_params[prop]['coeff'])
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()

    rewards = estimate_rewards_batch(properties,reward_params)
    # We only want to reward molecules with good docking_reward
    for this_mol, this_docking_reward in enumerate(rewards['prob_active']):
        rewards['TOTAL'][this_mol] = 1
        if this_docking_reward == 15 and rewards['scaffold_match'][this_mol] == 15:

            # randomly choose *one* second property to consider.
            # Also, allow for a 1% probability of accepting a "bad" molecule.
            prop_to_consider = props[np.random.choice(len(props),p=probabilities)]
            #prop_to_consider = props[np.random.choice(len(props))]
            if rewards[prop_to_consider][this_mol] == 15 or rng.random() >= 0.99:
                rewards['TOTAL'][this_mol] = 15
    return rewards

# =============================================================================
#           CALCULATORS FOR SINGLE MOLECULES
# =============================================================================
# Tests Scores and Rewards

def estimate_properties_one(molecule:str, properties_to_estimate:Dict) -> Dict: 
    """Estimates all properties for ONE mlecule

    Args:
        molecule (str): SMILES string for one molecule
        properties_to_estimate (Dict): Properties to estimate

    Returns:
        Dict: Estimated properties.
    """
    # Create the dictionary for properties values    
    properties = {}
    for prop_name in properties_to_estimate.keys():
        properties[prop_name] = []

    # Each 'molecule' is a SMILES string    
    mol = Chem.MolFromSmiles(molecule, sanitize=True)
    for prop_name, params in properties_to_estimate.items():
        if prop_name == 'prob_active' and params['model_type'] == "CHEMBERT":
            from properties.activity.CHEMBERT.chembert import SMILES_Dataset
            dataset = SMILES_Dataset([molecule])
            properties[prop_name] = params['predictor'].predict(dataset)
        else:
            prop = getattr(sys.modules[__name__], prop_name)
            this_prop = (prop(mol,**params))
            properties[prop_name].append(this_prop)

    return properties

def estimate_rewards_one(properties:Dict,reward_params:Dict) -> Dict:
    """ Estimates the rewards corresponding to each property, for ONE molecule.
        Assumes that the properties have already been calculated and stored
        into the 'properties' dictionary passed in. 

    Args:
        properties (Dict): The calculated properties
        reward_params (Dict): parameters for reward calculation

    Returns:
        Dict: Rewards for the molecule. One per property, plus a "TOTAL".
    """
    # Create the dictionary for rewards    
    rewards = {}
    rewards["TOTAL"] = 0
    
    # Now, we loop through each property and get their
    # respective reward, then sum into the TOTAL reward
    for prop_name in properties.keys():
        
        # Dynamically builds the name of the reward function to call
        prop = getattr(sys.modules[__name__], f"{prop_name}_reward")

        # For each value in this property, calculate the reward
        for prop_value in properties[prop_name]:

            this_reward = ( prop(prop_value, **reward_params[prop_name]) )
            rewards[prop_name] = this_reward
            rewards["TOTAL"] += this_reward
            
    return rewards

def estimate_capped_rewards_one(properties:Dict,reward_params:Dict) -> Dict:
    """ Estimates the rewards corresponding to each property, for ONE molecule.
        Assumes that the properties have already been calculated and stored
        into the 'properties' dictionary passed in. 

        In addtion, the "TOTAL" column here is capped so that the total reward is
        either 1 or 15.

        To get a reward of 15 the molecule must meet the `prob_active` and 
        `scaffold_match` requirements, plus the requirements for one extra property
        chosen at random with probabilities equal to the property coeff.
        
    Args:
        properties (Dict): The calculated properties
        reward_params (Dict): parameters for reward calculation

    Returns:
        Dict: Rewards for the molecule. One per property, plus a "TOTAL".
    """


    probabilities = []

    # This sets the list pr properties that will be considered beyond
    # prob_active ad scaffold match. So, we will need to remove those
    # two from the list.
    props = list(properties.keys())
    props.remove('prob_active')    # MANDATORY
    props.remove('scaffold_match') # MANDATORY

    # This will get the coeff of all the other properties and
    # transform into probabilities. The sum of all probabilities
    # must be 1.
    for prop in props:
        probabilities.append(reward_params[prop]['coeff'])
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()

    rewards = estimate_rewards_one(properties,reward_params)

    rewards['TOTAL'] = 1
    if rewards['prob_active'] == 15 and rewards['scaffold_match'] == 15:
        prop_to_consider = props[np.random.choice(len(props),p=probabilities)]
        if rewards[prop_to_consider] == 15 or rng.random() >= 0.99:
            rewards['TOTAL'] = 15
    return rewards
# =============================================================================
#           INDIVIDUAL PROPERTY CALCULATIONS
# =============================================================================

def prob_active(query_mol:rdkit.Chem.Mol, **kwargs) -> float:

    predictor = kwargs['predictor']
    predictor_type = kwargs['predictor_type']
    
    score = 0.0

    if query_mol is not None:
        #fp = GetMorganFingerprintAsBitVect(query_mol, 2)
        fp = RDKFingerprint(query_mol)
        #fp = [np.array(list(map(int,fp.ToBitString())),dtype='i1')]

        if predictor_type == 'classifier':
            # Careful here: make sure we have the correct classes!
            # This assumes 0 = INACTIVE, 1 = ACTIVE
            score = predictor.predict_proba([fp])[0][1]
        else:
            # Means it is a regressor
            score = predictor.predict([fp])[0]
    
    return score

# ---------------------------------------------------------------------
#
#                           SCAFFOLD MATCH
#
# ---------------------------------------------------------------------
class CompareQueryAtoms(rdFMCS.MCSAtomCompare):
    """This class defines a custom atom comparison to be used. It is called by
        FindMCS when the two molecules are compared.
    """
    def __call__(self, p, mol1, atom1, mol2, atom2):
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        match = True
        if ((a1.GetAtomicNum() != 0) and (a2.GetAtomicNum() != 0)) and (a1.GetAtomicNum() != a2.GetAtomicNum()):
             match = False
        elif (p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence()):
            match = False
        elif (p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2)):
            match = False
        elif (p.MatchFormalCharge and (not a1.HasQuery()) and (not a2.HasQuery()) and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2)):
            match = False
        elif p.RingMatchesRingOnly and not ( a1.IsInRing() == a2.IsInRing() ):
            match = False
        elif ((a1.HasQuery() or a2.HasQuery()) and (not a1.Match(a2))):
            match = False
        return match

def scaffold_match(query_mol:rdkit.Chem.Mol, **kwargs) -> float:
    """Checks if the scaffold from scaff_smi is 
        contained in the query_smi. This results in
        a percent match in the [0-1] (%) interval 

    Args:
        query_mol (rdkit.Chem.Mol): The query molecule

    Returns:
        float: The percent scaffold match [0,1].
    """
    params = rdFMCS.MCSParameters()
    
    params.AtomCompareParameters.CompleteRingsOnly   = True
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.AtomCompareParameters.MatchChiralTag      = False
    params.AtomCompareParameters.MatchFormalCharge   = False
    params.AtomCompareParameters.MatchIsotope        = False
    params.AtomCompareParameters.MatchValences       = False

    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly   = True
    params.BondCompareParameters.MatchFusedRings       = False
    params.BondCompareParameters.MatchFusedRingsStrict = False
    params.BondCompareParameters.MatchStereo           = False
    
    # Requests the use of the `CompareQueryAtoms` class (defined above) to compare atoms
    params.AtomTyper = CompareQueryAtoms()

    # No custom matching set for bonds. It will just use the `CompareAny`, which allows
    # bonds of any order to be compared.
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    scaffold = kwargs['scaffold']
    match = 0.0
    if query_mol is not None:
        maxMatch = scaffold.GetNumAtoms()
        match = rdFMCS.FindMCS([scaffold,query_mol],params).numAtoms / maxMatch
    return match

def synthetic_accessibility(query_mol:rdkit.Chem.Mol, **kwargs) -> float:
    """
    Gets a RDKit ROMol object and returns a reward proportional to its
    Synthetic Accessibility Score (SAS). The SAS is in the [1,10] interval
    where:
         1 == Very EASY to synthesize
        10 == Very HARD to synthesize.

    Here we use the SASCorer method developed by scientists at
    Novartis Institutes for BioMedical Research Inc.
    For details, see: http://www.doi.org/10.1186/1758-2946-1-8

    Args:
        query_mol (rdkit.Chem.Mol): Molecule to be scored

    Returns:
        float: The molecule's SA_Score
    """
    sa_score = 10.0
    if query_mol is not None:
        try:
            sa_score = sascorer.predict([query_mol])[0] 
        except:
            pass
    return sa_score

def drug_likeness(query_mol:rdkit.Chem.Mol, **kwargs) -> float:
    """Calculates a score based on drug likeness. Uses the method described in:

       _'Quantifying the chemical beauty of drugs'_
       Nature Chemistry volume 4, pages90–98(2012)
       https://doi.org/10.1038/nchem.1243

       As implemented in RDKit QED module, QED values are in the interval [0,1]:
	    0 == BAD  (all properties unfavourable) 
        1 == GOOD (all properties favourable)

    Args:
        mol (rdkit.Chem.ROMol): molecule to be evaluated

    Returns:
        float: Drug likeness score rounded to next integer
    """
    qed_score = 0.0
    if query_mol is not None:
        try:
            qed_score = QED.qed(query_mol)
        except:
            # RDKit gives exception when the molecules are weird. 
            # Here we just ignore them and pass a score of zero.
            pass
    return qed_score

# =============================================================================
#             REWARD CALCULATIONS
# =============================================================================
#
# The reinforcement algorithm works better if the total reward is a step function.
# So, every reward will be either 1 or 15, depending on being over a pre-defined
# threshold or not.

def prob_active_reward(prob_active:float, **kwargs) -> int:
    """Reward for predicted docking score. 
       This is the main reward: if this condition is not satisfied, the other
       rewards are not even taken into consideration when selecting molecules. 

    Args:
        prob_active (float): [description]

    Returns:
        int: [description]
    """

    reward = 1
    threshold = kwargs['threshold']
    predictor_type = kwargs['predictor_type']
    if predictor_type == 'classifier':
        if prob_active >= threshold:
            reward = 15
    elif prob_active <= threshold:
        reward = 15
    return reward


def scaffold_match_reward(scaffold_match:float, **kwargs) -> int:
    """Reward for scffold matching. The input is 
       a match number in the [0,1] interval, where

       0 == No atoms match
       1 == All atoms match.

    Args:
        scaffold_match (float): Percent scaffold match

    Returns:
        int: reward
    """
    reward = 1
    if scaffold_match >= kwargs['threshold']:
            reward = 15
    return reward

def synthetic_accessibility_reward(sa_score:float, **kwargs) -> int:
    """ Reward for synthetic accessibility score.
        SAS values are in the range [1,10], where:

         1 == Very EASY to synthesize
        10 == Very HARD to synthesize.

    Args:
        sa_score (float): The SAS score

    Returns:
        int: Reward
    """
    reward = 1
    if sa_score <= kwargs['threshold']:
            reward = 15
    return reward

def drug_likeness_reward(qed_score:float, **kwargs) -> int:
    """Reward for drug likeness. QED values are in the interval [0,1]:
	    0 == BAD  (all properties unfavourable) 
        1 == GOOD (all properties favourable)

        For details, see: http://www.doi.org/10.1038/nchem.1243

    Args:
        qed_score (float): The calculated QED score

    Returns:
        int: Calculated reward
    """
    reward = 1
    if qed_score >= kwargs['threshold']:
            reward = 15
    return reward

# =============================================================================
#             THRESHOLD ADJSTMENTS
# =============================================================================

def check_and_adjust_thresholds(predictions_cur:Dict, rewards_cur:Dict, reward_properties:Dict):
    """ Entry point for cheching and adjusting thresolds.

    Args:
        predictions_cur (Dict): Pre-calculated predictions for the properties
        rewards_cur (Dict): Pre-calculated rewards for the properties
        reward_properties (Dict): Definitions of properties
    """
    print("\nCHECKING AND AJUSTING THRESHOLDS:")
    print("-"*50)

    # First, we make sure that the propertis only have values if the 
    # molecule pass the 'mandatory' tests.
    props = list(predictions_cur.keys())
    props.remove('prob_active')
    props.remove('scaffold_match') # MANDATORY
    print("Total Rewards:")
    total_rew_np = np.array(rewards_cur['TOTAL'])
    print(f"There were {np.sum(total_rew_np >= 15)} molecules approved from a total of {len(total_rew_np)} generated.")
    #for ind, rew in enumerate(rewards_cur['TOTAL']):
    #     if rew < 15.0:
    #         # A TOTAL reward < 15 means the molecule was not approved.
    #         # We don't want those molecules to influence the learning, so we set
    #         # their properties to the base (worst) value before proceeding:
    #         for prop_name in props:
    #             # Dynamically create the function name to be called.
    #             prop = getattr(sys.modules[__name__], prop_name)

    #             # Sets the property to the base value
    #             # (The value given to an invalid molecule)
    #             this_prop = (prop(None,**reward_properties[prop_name]))
    #             predictions_cur[prop_name][ind] = this_prop

    for prop in reward_properties.keys():
        if reward_properties[prop]['adjust_threshold'] == True:
            reward_properties[prop]['moved_threshold'] = False
            print(prop)
            # Dynamically create the function name to call
            func = getattr(sys.modules[__name__], f"check_and_adjust_{prop}_threshold")
            func(predictions_cur[prop],reward_properties[prop])
    return

def check_and_adjust_prob_active_threshold(scores:List, reward_properties:Dict):
    """ Checks if enough molecules are being generated with prob_active below
        the threshold. If that is the case, tighten the threshold a bit further.

    Args:
        scores (List): Last generated docking scores
        reward_properties (Dict): Definitions and properties for prob_actives only
    """

    thresh_limit = reward_properties["threshold_limit"]
    threshold = reward_properties["threshold"]
    predictor_type = reward_properties["predictor_type"]
    pred = np.array(scores) 

    moved_threshold = False
    if predictor_type == 'classifier':
        # On a classifier, higher is better

        above_thr = np.sum(pred >= threshold)
        above_thr_percent = above_thr / len(pred)
        print(f"    --> Threshold    : {threshold:6.2f}")
        print(f"    --> n_mols ABOVE : {above_thr} ({100*above_thr_percent:4.1f}%)")

        # Adjust the threshold
        # If enough predictions are above the threshold, 
        # adjust threshold for the next round
        if above_thr_percent >= 0.3 and threshold < thresh_limit:
            # new_thr = threshold - reward_properties["threshold_step"]
            # reward_properties["threshold"] = new_thr

            # Changes the threshold to the 25% percentile
            reward_properties["threshold"] = min(thresh_limit, max(threshold, np.percentile(pred,75,interpolation='higher')))
            moved_threshold = True

            print((f"    --> ADJUSTING DOCKING SCORE THRESHOLD TO " 
                        f"{reward_properties['threshold']:.5f}, " 
                        f"BEGINNING NEXT ITERATION."))
    
    else:
        # This is a regressor, and lower is better

        below_thr = np.sum(pred <= threshold)
        below_thr_percent = below_thr / len(pred)
        print(f"    --> Threshold    : {threshold:6.2f}")
        print(f"    --> n_mols BELOW : {below_thr} ({100*below_thr_percent:4.1f}%)")

        # Adjust the threshold
        # If enough predictions are below the threshold, 
        # adjust threshold for the next round
        if below_thr_percent >= 0.3 and threshold > thresh_limit:

            # Changes the threshold to the 25% percentile
            reward_properties["threshold"] = max(thresh_limit, min(threshold, np.percentile(pred,25,interpolation='lower')))
            moved_threshold = True

            print((f"    --> ADJUSTING DOCKING SCORE THRESHOLD TO " 
                        f"{reward_properties['threshold']:.5f}, " 
                        f"BEGINNING NEXT ITERATION."))

    reward_properties['moved_threshold'] = moved_threshold
    return
    
def check_and_adjust_scaffold_match_threshold(matches:List, reward_properties:Dict):
    """ Checks if enough molecules are being generated with scaffold_match above
        the threshold. If that is the case, tighten the threshold a bit further.

    Args:
        matches (List): Last generated scaffold_matches
        reward_properties (Dict): Definitions and properties for scaffold_match only
    """

    thresh_limit = reward_properties["threshold_limit"]
    threshold = reward_properties["threshold"]
    step = reward_properties["threshold_step"]
    pred = np.array(matches)

    above_thr = np.sum(pred >= threshold)
    above_thr_percent = above_thr / len(pred)
    print(f"    --> Threshold    : {threshold:6.2f}")
    print(f"    --> n_mols ABOVE : {above_thr} ({above_thr_percent:4.1%})")

    # Adjust the threshold
    # If enough predictions are above the threshold, 
    # adjust threshold for the next round
    moved_threshold = False
    if above_thr_percent >= 0.3 and threshold < thresh_limit:
        # LARGER IS BETTER
        # This allows for large jumps in the threshold, if needed
        new_thr = np.percentile(pred,75, interpolation='lower')
        if (new_thr - threshold) <  step: new_thr = threshold + step
        reward_properties["threshold"] = min(thresh_limit, new_thr)
        moved_threshold = True

        print((f"    --> ADJUSTING SCAFFOLD_MATCH THRESHOLD TO " 
                    f"{reward_properties['threshold']:5.2f}, " 
                    f"BEGINNING NEXT ITERATION."))
    reward_properties['moved_threshold'] = moved_threshold
    return

def check_and_adjust_synthetic_accessibility_threshold(sa_scores:List, reward_properties:Dict):
    """ Checks if enough molecules are being generated with synthetic_accessibility below
        the threshold. If that is the case, tighten the threshold a bit further.

    Args:
        sa_scores (List): Last generated sa_scores
        reward_properties (Dict): Definitions and properties for sa_scores only
    """

    thresh_limit = reward_properties["threshold_limit"]
    threshold = reward_properties["threshold"]
    pred = np.array(sa_scores)

    below_thr = np.sum(pred <= threshold)
    below_thr_percent = below_thr / len(pred)
    print(f"    --> Threshold    : {threshold:6.2f}")
    print(f"    --> n_mols BELOW : {below_thr} ({below_thr_percent:4.1%})")

    # Adjust the threshold
    # If enough predictions are below the threshold, 
    # adjust threshold for the next round

    moved_threshold = False
    if below_thr_percent >= 0.3 and threshold > thresh_limit:
        # SMALLER IS BETTER
        # new_thr = threshold - reward_properties["threshold_step"]
        #reward_properties["threshold"] = max(thresh_limit,new_thr)

        # Changes the threshold to the 25% percentile
        reward_properties["threshold"] = max(thresh_limit, 
                                             min(threshold, 
                                                 np.percentile(pred,25,interpolation='higher')))

        print((f"    --> ADJUSTING SA_SCORES THRESHOLD TO " 
                    f"{reward_properties['threshold']:5.2f}, " 
                    f"BEGINNING NEXT ITERATION."))
        moved_threshold = True
    elif below_thr_percent <= 0.01 and threshold > thresh_limit:
        # The threshold is too tight, resulting in NO molecules getting accepted.
        # Try to adjust to rescue the 'least bad' molecule.
        print(f"    --> SAScore threshold is too tight. Adjusting it to rescue some molecules.")
        reward_properties["threshold"] = min(threshold + reward_properties["threshold_step"],
                                             10)
        moved_threshold = True
    
    if moved_threshold:
        print((f"    --> ADJUSTING SASCore THRESHOLD TO " 
                    f"{reward_properties['threshold']:5.2f}, " 
                    f"BEGINNING NEXT ITERATION."))

    reward_properties['moved_threshold'] = moved_threshold
    return


def check_and_adjust_drug_likeness_threshold(qedscores:List, reward_properties:Dict):
    """ Checks if enough molecules are being generated with drug_likeness below
        the threshold. If that is the case, tighten the threshold a bit further.

    Args:
        sa_scores (List): Last generated sa_scores
        reward_properties (Dict): Definitions and properties for sa_scores only
    """

    thresh_limit = reward_properties["threshold_limit"]
    threshold = reward_properties["threshold"]
    pred = np.array(qedscores)

    above_thr = np.sum(pred >= threshold)
    above_thr_percent = above_thr / len(pred)
    print(f"    --> Threshold    : {threshold:6.2f}")
    print(f"    --> n_mols ABOVE : {above_thr} ({above_thr_percent:4.1%})")

    # Adjust the threshold. Use only approved molecules in the
    # calculation of the next values.
    pred_approved = pred[ pred >= threshold ]
    moved_threshold = False

    if above_thr_percent >= 0.3 and threshold < thresh_limit:
        # LARGER IS BETTER

        # Moves the threshold by small steps
        #new_thr = threshold + reward_properties["threshold_step"]
        #reward_properties["threshold"] = min(thresh_limit, new_thr)

        # Moves the threshold by big jumps
        # Changes the threshold to the 75% percentile

        # We also include the step size here to make sure 
        # the threshold is never lower than the step size.
        reward_properties["threshold"] = min(thresh_limit, 
                                             max(threshold, 
                                                 np.percentile(pred_approved,75,interpolation='lower'), 
                                                 reward_properties["threshold_step"]))
        moved_threshold = True

    elif above_thr_percent <= 0.01 and threshold < thresh_limit:
        # The threshold is too tight, resulting in NO molecules getting accepted.
        # Try to adjust to rescue the 'least bad' molecule.
        print(f"    --> Drug likeness threshold is too tight. Adjusting it down to rescue some molecules.")
        reward_properties["threshold"] = max(threshold - reward_properties["threshold_step"],
                                             reward_properties["threshold_step"])
        moved_threshold = True
    
    if moved_threshold:
        print((f"    --> ADJUSTING DRUG LIKENESS THRESHOLD TO " 
                    f"{reward_properties['threshold']:5.2f}, " 
                    f"BEGINNING NEXT ITERATION."))

    reward_properties['moved_threshold'] = moved_threshold
    return

def all_converged(all_reward_properties:Dict) -> bool:
    all_converged = True
    print('='*75)
    print(f"{'PROPERTIES THRESHOLD CONVERGENCE':^75s}")
    print('='*75)
    print(f"{'PROPERTY':<30s} \t {'LIMIT':>10s} \t {'CURRENT':>10s} \t {'CONVERGED?':10s}")
    print('-'*75)
    for key in all_reward_properties.keys():
        prop = all_reward_properties[key]
        if prop['adjust_threshold'] == True:
            this_converged = (prop['threshold'] == prop['threshold_limit'])
            print(f"{key:<30s} \t {prop['threshold_limit']:10.2f} \t {prop['threshold']:10.2f} \t {str(this_converged):>10}")
            if not this_converged: all_converged = False
    print('='*75)

    return all_converged
