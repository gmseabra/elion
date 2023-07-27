"""Calculates the similarity score between two molecules.
"""

from pathlib import Path

# Chemistry
import rdkit
from rdkit import Chem, DataStructs

# Suppress RDKit output
from rdkit import RDLogger
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

from properties.Property import Property

class SimilarityScore(Property):
    """
        Similarity Score.
        
        Given a molecule and a scaffold, this property
        returns a percent coverage of the scaffold by the molecule.

        Checks if the scaffold from scaff_smi is 
        contained in the query_smi. This results in
        a percent match in the [0-1] (%) interval 

    """

    CITATION = (f" \"RDKit: Open-source cheminformatics version {rdkit.__version__} "
                 "(www.rdkit.org)\"")

    def __init__(self, prop_name, **kwargs):

        # Initialize super
        super().__init__(prop_name, **kwargs)

        # Reference Molecule
        if 'reference_smi' in kwargs.keys():
            reference_smi = kwargs['reference_smi']
            reference_mol = Chem.MolFromSmiles(reference_smi)
            if not reference_mol:
                self.bomb_input(("ERROR while reading reference molecule:\n"
                                f"Cannot convert {reference_smi} to RDKit Mol object."))
            
            print(f"Using molecule '{reference_smi}' as refence for similarity. ")
            self.reference = reference_mol
        else:
            self.bomb_input("ERROR: Reference molecule not specified.")

        # RegionSelector
        # Can be: 'molecule', 'murcko' or 'generic_murcko

        # By default, the region selector is the molecule itself
        self.region_selector = lambda x, *args: (x,) + args if args else x

        # If indicated, use MurckoScaffold or GenericMurckoScaffold
        if 'region_selector' in kwargs.keys():
            if kwargs['region_selector'] == 'murcko':
                # Use the Murcko Scaffold for similarity measures
                self.region_selector = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol
                print("Using the Murcko Scaffold for similarity measures. ")
                
            elif kwargs['region_selector'] == 'generic_molecule':
                # This just replaces all atoms for carbon and all bonds for single bonds
                self.region_selector = Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric
                print(("Using the Generic molecule for similarity measures. \n"
                       "(All atoms are carbons and all bonds are single bonds)"))
                
            elif kwargs['region_selector'] == 'generic_murcko':
                # First get the Murcko Scaffold and then make it generic
                self.region_selector = self.generic_murcko_scaffold
                print("Using the Generic Murcko Scaffold for similarity measures. ")

            else:
                self.bomb_input(f"ERROR: Region Selector '{kwargs['region_selector']}' not recognized.")
        
        # Fingerprint type
        # Deafault is 'rdkit'
        self.fingerprinter = Chem.rdFingerprintGenerator.GetRDKitFPGenerator()
        if 'fingerprint' in kwargs.keys():
            if kwargs['fingerprint'] == 'morgan':
                radius = kwargs['radius'] if 'radius' in kwargs.keys() else 2
                self.fingerprinter = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius)

        # Metric
        # Only Tanimoto is supported for now
        self.metric = DataStructs.TanimotoSimilarity

        # Finally, prepare the reference molecule
        self.reference_fp = self.fingerprinter.GetFingerprint(self.region_selector(self.reference))

    def generic_murcko_scaffold(self, mol):
        # First get the Murcko Scaffold and then make it generic
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return scaffold
    
    def predict(self,mols, **kwargs):
        """
            Args:
                mol (rdkit.Chem.Mol or list): The query molecule(s)

            Returns:
                list(float): The similarity score [0,1].
        """
        _mols, similarity_scores = [], []
        _mols.extend(mols)
        reference = self.reference

        for query_mol in _mols:
            score = 0.0
            try:
                score = self.metric(self.reference_fp, 
                                    self.fingerprinter.GetFingerprint(self.region_selector(query_mol)))
            except:
                # RDKit gives exception when the molecules are weird. 
                # Here we just ignore them and pass a score of -1.
                pass
            similarity_scores.append(score)
        return similarity_scores

    
