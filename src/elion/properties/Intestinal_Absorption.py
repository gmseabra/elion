# Predicts ADMET properties for molecules
import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem

from properties.Property import Property


class Intestinal_Absorption(Property):
    """ Estimation of the fraction of compound absorbed at 100mg dose
        using SimulationsPlus ADMET Predictor.

        ** 89% of the WDI drugs fall into %Fa > 80% cutoff **

        Note that AMDET Predictor is a commercial product, and must be
         purchased separately. This class assumes that:

        (i)    ADMET Predictor is installed in the system.
        (ii)   A custom SimHIA-100.hia file exists in the 
                `src/elion/properties/admet` folder, modified
                 to use a 100 mg dose.
        (iii)  The ADMET Predictor executable is specified in the
                config file in the entry 'RunAP_executable'. 
                That would usually be the full path to the `RunAP.sh` executable.
    """

    CITATION = (" ADMET Predictor v11,\n"
                "  https://www.simulations-plus.com/software/admet-predictor/")

    DOSE = 100.0  # mg

    def __init__(self, prop_name, **kwargs):
        # Initialize super
        super().__init__(prop_name, **kwargs)
        if 'RunAP_executable' not in kwargs:
            msg = "ERROR: ADMET Predictor executable not specified in config file."
            self.bomb_input(msg)
        else:
            self.executable = Path(kwargs['RunAP_executable'])
        if not self.executable.is_file():
            msg = (f"ERROR: ADMET Predictor executable not found at {self.executable.absolute()}.\n"
                    "Please check the path to the executable in the config file.")
            self.bomb_input(msg)
        else:
            print("  Executable file: ", self.executable)

        # Modified Parameters File for 100 mg dose
        self.params_file = Path(__file__).parent / 'admet' / 'SimHIA-100.hia'

    def predict(self, mols, **kwargs):
        """Predict fraction of the compound absorbed (%Fa) property for molecules.

           At the moment, there is no interface to ADMET Predictor, so we 
           connect to it via the command line. This means that we need to
           save the molecules to a file, run ADMET Predictor, and then read
           the results from the output file.

        Args:
            mols: RDKit Mol or list of RDKit Mols

        Returns:
            absorption: list of predicted fraction absorbed values for each molecule
        """

        _mols, absorbed = [], []
        _mols.extend(mols)
        smiles = [Chem.MolToSmiles(x) for x in _mols]

        smiles_file = Path(tempfile.NamedTemporaryFile(suffix='.smi', delete=False).name)
        output_file = smiles_file.with_suffix('.dat')

        with open(smiles_file,'w', encoding='utf-8') as nf:
            for seq, smi in enumerate(smiles):
                nf.write(f"{smi}\tGen-{seq}\n")
                        
        # Now, runs ADMET-Predictor with this file
        _ = subprocess.run([self.executable,
                            "-t", "SMI",
                            smiles_file,
                            "-m","SimFaFb",
                            "-SimHIA_hia", self.params_file,   
                            "-N","16",
                            "-out", Path(output_file.parent,output_file.stem)],
                           capture_output=True,
                           check=False)

        # now we read the output file to extract the ADMET Risk values
        absorbed_column = None
        with open(output_file, 'r', encoding='utf-8') as f:
            col_header = f"%Fa_hum-{self.DOSE:.1f}"
            for line in f.readlines():
                if col_header in line:
                    absorbed_column = line.split("\t").index(col_header)
                    continue                
                absorbed.append(float(line.split("\t")[absorbed_column]))
        # Finally, we delete the temporary & junk files created by ADMET Predictor
        junk = [smiles_file, output_file]
        junk.extend(list(Path.cwd().glob('flex*.log')))
        junk.extend(list(smiles_file.parent.glob('ADMET*.log')))
        for j in junk:
            Path(j).unlink(missing_ok=True)
            
        return absorbed
