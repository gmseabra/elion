# Predicts ADMET properties for molecules
import subprocess
import tempfile
from properties.Property import Property

class ADMET_Predictor(Property):
    """Estimation of ADMET properties using SimulationsPlus ADMET Predictor.

       Note that AMDET Predictor is a commercial product, and must be
         purchased separately. This class assumes that ADMET Predictor
         is installed in the system, and that the executable is in the
         
    """


    def predict_admet(smiles_file, admet_predictor, output_stem, keep_smi):
        """Predict ADMET properties for molecules in file <filename>

        Args:
            smiles_file (file): A CSV SMILES file from Elion, that MUST have a
                                named "SMILES". It *may* also have a "Name" column.
                                All other columns will be ignored.
        """

        # Fist, convert the SMILES file to the format required
        # by ADMET predictor
        convert_smiles(smiles_file, output_stem)


        return

    
    def predict(self,mols, **kwargs):
        _mols, admet_properties = [], []
        _mols.extend(mols)
        smiles = [Chem.MolToSmiles(x) for x in _mols]

        smiles_file = tempfile.NamedTemporaryFile(delete=False)
        with open(smiles_file,'w') as nf:
            for seq, smi in enumerate(smiles):
                nf.write(f"{smi}\tGen-{seq}\n")

        # Now, runs ADMET-Predictor with this file
        _ = subprocess.run([admet_predictor,
                            "-t", "SMI",
                            f"{output_stem}.smi",
                            "-m","TOX,GLB,SimFaFb",
                            "-N","16",
                            "-out",output_stem])
        Path(smiles_file).unlink()


        return vina_scores
