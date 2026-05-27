import os
import sys
from collections import deque
# import openbabel as ob
# import sys
# chimera_path = sys.path
# sys.path += ['/blue/lic/huangzihang/repos/miniconda3/envs/GIGN/lib/python39.zip', '/blue/lic/huangzihang/repos/miniconda3/envs/GIGN/lib/python3.9', '/blue/lic/huangzihang/repos/miniconda3/envs/GIGN/lib/python3.9/lib-dynload', '/home/huangzihang/.local/lib/python3.9/site-packages', '/blue/lic/huangzihang/repos/miniconda3/envs/GIGN/lib/python3.9/site-packages']
# sys.path = ['', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python27.zip', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7/plat-linux2', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7/lib-tk', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7/lib-old', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7/lib-dynload', '/blue/lic/huangzihang/.conda/envs/new_binding_affinity_27/lib/python2.7/site-packages']
import subprocess
import itertools

# sys.path = chimera_path
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.PDBIO import Select


dataset_dir = sys.argv[1]
vs_dataset_dir = os.path.join(dataset_dir, "Dataset_VS")

os.chdir(vs_dataset_dir)

pdb_codes = [pdb_code for pdb_code in os.listdir(".")]

counter = 0
num_complx = len(pdb_codes)


for pdb_code in pdb_codes:
    prot_path = os.path.join(vs_dataset_dir, pdb_code)
    os.chdir(prot_path)

    counter +=1
    print("===================================================================")
    print("{}:   complex {} (out of {})".format(pdb_code, counter, num_complx))
    #print ""

    lig_mol2 = "{0}_ligand.mol2".format(pdb_code)
    lig_pdb = "{0}_ligand.pdb".format(pdb_code)
    prot_pdb = "{0}_protein.pdb".format(pdb_code)
    cmplx_pdb = "{0}_complex.pdb".format(pdb_code)

    # intermediate files
    prot_wt_altloc = "{0}_protein_wt_altloc.pdb".format(pdb_code)

    #========================================================================

    # remove altloc records, except altloc "A"
    # also remove water molecules
    pdb_parser = PDBParser()
    s = pdb_parser.get_structure('my_pdb', prot_pdb)

    class NotDisordered(Select):
        def accept_atom(self, atom):
            residue_id = atom.get_parent().get_id()
            hetfield = residue_id[0]

            isNotWater = hetfield[0]!="W"   # i.e. water
            isOrderedOrFirstAltloc = not atom.is_disordered() or atom.get_altloc() == 'A'
            
            return isNotWater and isOrderedOrFirstAltloc
    
    io = PDBIO()
    io.set_structure(s)
    io.save(prot_wt_altloc, select=NotDisordered())

    #========================================================================

    # remove the altloc identifier "A" from column 17
    subprocess.call(["sed", "-i", "-e", 's/./ /17', prot_wt_altloc])

    #========================================================================

    if os.path.exists(lig_mol2) and not os.path.exists(lig_pdb):

        # convert ligand from MOL2 to PDB format
        try:
            # convert ligand from MOL2 to PDB format
            obConversion = ob.OBConversion()
            obConversion.SetInAndOutFormats("mol2", "pdb")
            mol = ob.OBMol()
            obConversion.ReadFile(mol, lig_mol2)
            obConversion.WriteFile(mol, lig_pdb)
        except:
            print("*" * 20 + pdb_code + "*" * 20)
            os.chdir("..")
            continue

    #========================================================================

    # concatenate the protein and ligand files
    print('-----------------debug6 os.getcwd(): %s' % os.getcwd())
    print('-----------------debug6 cmplx_pdb: %s' % cmplx_pdb)

    with open(cmplx_pdb, 'w') as outfile:  # 'w' not 'a': avoid corrupt output on retry
        # use a deque to keep track of the last two lines added;
        # it keeps both last line containing an atom, and the TER line
        last_line = deque(['_', '_'], maxlen=2)

        # print('-----------------debug3 prot_wt_altloc: %s' % prot_wt_altloc)
        with open(prot_wt_altloc, 'r') as infile:
            for line in infile:

                isEND = line.startswith("END")
                isTER = line.startswith("TER")
                if line=="END\n":
                    isHydrogen = False
                else:
                    isHydrogen = not isTER and len(line) > 77 and line[77]=="H"

                if not isEND and not isHydrogen and not isTER:
                    last_line.append(line)
                    outfile.write(line)

                elif isTER:
                    if len(last_line) < 2 or last_line[0] == '_':
                        last_line.append(line)

                    if len(last_line) < 2:
                        # Not enough context to build a proper TER line; write a bare one
                        outfile.write("TER\n")
                        continue

                    if last_line[1]=="TER\n":
                        ter_serial = deque(itertools.islice(last_line[0], 6, 11))
                    else:
                        ter_serial = deque(itertools.islice(last_line[1], 6, 11))

                    ter_serial_str = "%5s" % str(int(''.join(ter_serial).strip()) + 1)
                    
                    if last_line[1]=="TER\n":
                        ter_resName = deque(itertools.islice(last_line[0], 17, 20))
                    else:
                        ter_resName = deque(itertools.islice(last_line[1], 17, 20))

                    ter_resName_str = ''.join(ter_resName)

                    if last_line[1]=="TER\n":
                        chain_id = last_line[0][21]
                    else:
                        chain_id = last_line[1][21]

                    chain_id_str = ''.join(chain_id)

                    # column 27 is the insertion code, e.g. in 1BCU.pdb
                    if last_line[1]=="TER\n":
                        ter_resSeq = deque(itertools.islice(last_line[0], 22, 27))
                    else:
                        ter_resSeq = deque(itertools.islice(last_line[1], 22, 27))
                    
                    ter_resSeq_str = ''.join(ter_resSeq)

                    TER_line = "TER   " + ter_serial_str + " "*6 +  \
                                ter_resName_str + " " + chain_id +  \
                                ter_resSeq_str + "\n"

                    outfile.write(TER_line)

                elif isHydrogen:
                    #print "10 === " 
                    #print last_line
                    pass

                elif isEND:
                    break

        # sample last two lines (when PDB file is downloaded from RCSB):

        # ATOM   5629  OXT VAL B 377     112.271  49.718  -3.472  1.00 25.75           O  
        # TER    5630      VAL B 377 

        # However, when the PDB file is taken from DeepChem's copy of PDBbind,
        # the second line has only "TER" without anymore information.

        # renumber the ligand, before concatenating it to the protein
        # https://stackoverflow.com/questions/7367550/redirect-subprocess-to-a-variable-as-a-string
        # first find the last atom serial number in the protein

        # Determine last protein atom serial number.
        # last_line is a maxlen=2 deque; [0] is older, [1] is newer.
        # After the loop, [1] is the TER line (or last ATOM if no TER),
        # and [0] is the last ATOM line before it.
        last_atom_line = None
        for candidate in reversed(last_line):
            if candidate not in ('_', 'TER\n') and not candidate.startswith('TER'):
                last_atom_line = candidate
                break
        if last_atom_line is None:
            raise ValueError("Could not find last ATOM line in {}".format(prot_wt_altloc))
        last_prot_serial = int(last_atom_line[6:11].strip()) + 1


        # also add chain identifier "y" to the ligand

        with open(lig_pdb, 'r') as infile:
            atom_counter = 0
            for line in infile:
                if line.startswith(("ATOM", "HETATM")) and line[77]!="H":
                    # columns 7-11 are serial number of atom
                    old_lig_serial = int(line[6:11].strip())
                    new_lig_serial = last_prot_serial + old_lig_serial
                    new_lig_serial_str = "%5s" % (str(new_lig_serial))                    

                    last_col = line[76:78].strip()
                    atom_symbol = None

                    if len(last_col) > 0:
                        atom_symbol = last_col
                    else:
                        atom_symbol = line[12:16].strip()[0]

                    atom_counter += 1
                    atom_name_number = atom_symbol + str(atom_counter)
                    atom_name_number_str = None

                    if len(atom_name_number) < 4:
                        atom_name_number_str = " " + "%-3s" % atom_name_number
                    else:
                        atom_name_number_str = "%4s" % atom_name_number

                    lig_ResName = "LIG"

                    outfile.write("HETATM" + new_lig_serial_str + " " +  \
                                    atom_name_number_str + " " +  \
                                    lig_ResName + " " + 'y' + line[22:])

        outfile.write("END")

    #========================================================================

    os.chdir("..")