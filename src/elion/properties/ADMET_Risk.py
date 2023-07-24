# Predicts ADMET properties for molecules
import subprocess
import tempfile
from pathlib import Path
from properties.Property import Property

class ADMET_Risk(Property):
    """Estimation of ADMET Risk property using SimulationsPlus ADMET Predictor.

       Note that AMDET Predictor is a commercial product, and must be
         purchased separately. This class assumes that ADMET Predictor
         is installed in the system, and that the executable is in the
         
    """

    def __init__(self, prop_name, **kwargs):
        # Initialize super
        super().__init__(prop_name, **kwargs)
        self.executable = Path(kwargs['executable'])
        if not self.executable.is_file():
            print(f"ERROR: ADMET Predictor executable not found at {self.executable.absolute()}")
            bomb_input("Please check the path to the executable in the config file.")

    def predict(self, mols, **kwargs):
        """Predict ADMET Risk property for molecules.

           At the moment, there is no interface to ADMET Predictor, so we 
           connect to it via the command line. This means that we need to
           save the molecules to a file, run ADMET Predictor, and then read
           the results from the output file.

        Args:
            mols: RDKit Mol or list of RDKit Mols
        """

        _mols, admet_risk = [], []
        _mols.extend(mols)
        smiles = [Chem.MolToSmiles(x) for x in _mols]

        smiles_file = tempfile.NamedTemporaryFile(delete=False)
        with open(smiles_file,'w', encoding='utf-8') as nf:
            for seq, smi in enumerate(smiles):
                nf.write(f"{smi}\tGen-{seq}\n")

        # Now, runs ADMET-Predictor with this file
        _ = subprocess.run([admet_predictor,
                            "-t", "SMI",
                            f"{output_stem}.smi",
                            "-m","TOX,GLB,SimFaFb",
                            "-N","16",
                            "-out",output_stem],
                           check=False)
        Path(smiles_file).unlink()


        return vina_scores
