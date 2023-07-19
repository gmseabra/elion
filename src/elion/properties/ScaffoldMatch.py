from pathlib import Path

# Chemistry
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS

from properties.Property import Property

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

class ScaffoldMatch(Property):
    """
        Calculator for Scaffold Match.
        
        Given a molecule and a scaffold, this property
        returns a percent coverage of the scaffold by the molecule.

        Checks if the scaffold from scaff_smi is 
        contained in the query_smi. This results in
        a percent match in the [0-1] (%) interval 

    """


    def __init__(self, prop_name, **kwargs):

        # Initialize super
        super().__init__(prop_name, **kwargs)
        
        if 'scaffold_file' in kwargs.keys():
            template_smarts_file = Path(kwargs['scaffold_file'])
            if not template_smarts_file.is_file():
                msg = ("ERROR while loading template file:\n"
                      f"File {template_smarts_file.absolute()} doesn't exist.")
                self.bomb_input(msg)

            
            print(f"\nLoading scaffold from {template_smarts_file.absolute()}. ")
            with open(template_smarts_file,'r') as tf:
                template = tf.readline().strip()
            template = Chem.MolFromSmarts(template)


            # This prints info, but also forces the info about rings to be calculated.
            # It is necessary because (a bug?) in RDKit that does not calculate the
            # infor about rings until they are requested (or printed)
            print("\tAtom  AtNum  InRing?  Arom?")
            for idx, atom in enumerate(template.GetAtoms()):
                print(f"\t{idx:>4d}  {atom.GetAtomicNum():5d}  {str(atom.IsInRing()):>7}  {str(atom.GetIsAromatic()):>5}")
            self.scaffold = template
            print(f"  Scaffold: {Chem.MolToSmiles(self.scaffold)}")
            print(f"  Scaffold has {self.scaffold.GetNumAtoms()} atoms.")
            
        # Sets up the parameters
        self.params = rdFMCS.MCSParameters()

        self.params.AtomCompareParameters.CompleteRingsOnly   = True
        self.params.AtomCompareParameters.RingMatchesRingOnly = True
        self.params.AtomCompareParameters.MatchChiralTag      = False
        self.params.AtomCompareParameters.MatchFormalCharge   = False
        self.params.AtomCompareParameters.MatchIsotope        = False
        self.params.AtomCompareParameters.MatchValences       = False
        self.params.BondCompareParameters.RingMatchesRingOnly   = True
        self.params.BondCompareParameters.CompleteRingsOnly     = True
        self.params.BondCompareParameters.MatchFusedRings       = False
        self.params.BondCompareParameters.MatchFusedRingsStrict = False
        self.params.BondCompareParameters.MatchStereo           = False
        
        # Requests the use of the `CompareQueryAtoms` class (defined above) to compare atoms
        self.params.AtomTyper = CompareQueryAtoms()

        # No custom matching set for bonds. It will just use the `CompareAny`, which allows
        # bonds of any order to be compared.
        self.params.BondTyper = rdFMCS.BondCompare.CompareAny
        
    def predict(self,mols, **kwargs):
        """
            Args:
                mol (rdkit.Chem.Mol or list): The query molecule(s)

            Returns:
                list(float): The percent scaffold match [0,1].
        """
        _mols, scaffold_match = [], []
        _mols.extend(mols)
        scaffold = self.scaffold

        for query_mol in _mols:
            match = 0.0
            try:
                maxMatch = scaffold.GetNumAtoms()
                match = rdFMCS.FindMCS([scaffold,query_mol],self.params).numAtoms / maxMatch
            except:
                # RDKit gives exception when the molecules are weird. 
                # Here we just ignore them and pass a score of -1.
                pass
            scaffold_match.append(match)
        return scaffold_match

    
