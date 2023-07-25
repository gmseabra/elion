# Predicts ADMET properties for molecules
import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem

from properties.Property import Property


class ADMET_Risk(Property):
    """ Estimation of ADMET Risk property using SimulationsPlus ADMET Predictor.

        Note that AMDET Predictor is a commercial product, and must be
         purchased separately. This class assumes that ADMET Predictor
         is installed in the system.

        It is expected that the ADMET Predictor executable is specified in the
            config file in the entry 'RunAP_executable'. That would usually be the
            full path to the `RunAP.sh` executable.
    """

    def __init__(self, prop_name, **kwargs):
        # Initialize super
        super().__init__(prop_name, **kwargs)
        print(kwargs)
        if 'RunAP_executable' not in kwargs:
            msg = "ERROR: ADMET Predictor executable not specified in config file."
            self.bomb_input(msg)
        else:
            self.executable = Path(kwargs['RunAP_executable'])
        if not self.executable.is_file():
            msg = (f"ERROR: ADMET Predictor executable not found at {self.executable.absolute()}.\n"
                    "Please check the path to the executable in the config file.")
            self.bomb_input(msg)

    def predict(self, mols, **kwargs):
        """Predict ADMET Risk property for molecules.

           At the moment, there is no interface to ADMET Predictor, so we 
           connect to it via the command line. This means that we need to
           save the molecules to a file, run ADMET Predictor, and then read
           the results from the output file.

        Args:
            mols: RDKit Mol or list of RDKit Mols

        Returns:
            admet_risk: list of ADMET Risk values for each molecule
        """

        _mols, admet_risk = [], []
        _mols.extend(mols)
        smiles = [Chem.MolToSmiles(x) for x in _mols]

        smiles_file = Path(tempfile.NamedTemporaryFile(suffix='.smi', delete=False).name)
        output_file = smiles_file.with_suffix('.dat')

        with open(smiles_file,'w', encoding='utf-8') as nf:
            for seq, smi in enumerate(smiles):
                nf.write(f"{smi}\tGen-{seq}\n")
                        
        # Now, runs ADMET-Predictor with this file
        # "-m","TOX,GLB,SimFaFb",
        _ = subprocess.run([self.executable,
                            "-t", "SMI",
                            smiles_file,
                            "-m","GLB",
                            "-N","16",
                            "-out", Path(output_file.parent,output_file.stem)],
                           capture_output=True,
                           check=False)

        # now we read the output file to extract the ADMET Risk values
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if "SMILES" in line:
                    continue
                admet_risk.append(float(line.split("\t")[9]))
        # Finally, we delete the temporary & junk files created by ADMET Predictor
        junk = [smiles_file, output_file]
        junk.extend(list(Path.cwd().glob('flex*.log')))
        junk.extend(list(smiles_file.parent.glob('ADMET*.log')))
        for j in junk:
            Path(j).unlink(missing_ok=True)
            
        return admet_risk
