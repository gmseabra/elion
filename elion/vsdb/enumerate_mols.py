"""
Routines to expand the database by enumerating isomers
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from tqdm.auto import tqdm

def enumerate_stereoisomers(df, maxIsomers=10):
    """
    For the molecules whith >1 possible stereoisomers, generate all
    possibilities and updates the DataFrame accordingly. The newly generated
    molecules will be on the bottom of the df, and the original molecules
    removed.

    NOTE: At the end, there may be duplicate InChI Keys. In my experience, this
          happens when there is an sp2 nitrogen tht can get cis or trans on the 
          double bond. Although the SMILES notation can distinguish it, the
          InChI Keys are the same.

          I still could not verify if the problem is the InChI definition or the RDKit
          implementation
    """
    opts = StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=maxIsomers, onlyUnassigned=True)
    
    if 'n_stereo' not in df.columns:
        tqdm.pandas(desc="Calculating the number of stereoisomers:")
        df['n_stereo'] = df['ROMol'].progress_apply(Chem.EnumerateStereoisomers.GetStereoisomerCount)
        
    to_expand = df.index[ df['n_stereo'] > 1 ]

    # This column will hold the stereoisomer number. It is important later for renaming the molecules
    df["Stereoisomer"] = 1

    print(f"{len(to_expand)} molecules may have > 1 possible stereoisomers.")
    new_mols_added = 0
    for record in tqdm(to_expand, desc="Enumerating stereoisomers"): 
        m = df.loc[record]['ROMol']
        #print(Chem.MolToSmiles(m))
        isomers = tuple(EnumerateStereoisomers(m, options=opts))
        #print(f"Molecule has {len(isomers)} isomers.")
   
        
        for num, iso in enumerate(isomers):
            new_rec = df.loc[record].copy()
            new_rec['Stereoisomer'] = num + 1
            new_rec['ROMol']        = iso
            new_rec['SMILES']       = Chem.MolToSmiles(iso)
            new_rec['InChI Key']    = Chem.inchi.MolToInchiKey(iso)
            df = df.append(new_rec,ignore_index=True)
            new_mols_added = new_mols_added + 1
        
    # Now, drop the original records
    df = df.drop(to_expand)
    new_mols_added = new_mols_added - len(to_expand)
    print(f"Added {new_mols_added} new molecules to the DataFrame.")
    
    return df.drop(columns=['n_stereo']).reset_index(drop=True)


def enumerate_ionization_states(df, min_ph=6.4, max_ph=8.4, pka_precision=1.0):
    """
    *************************** WARNING *****************************
    This can lead to a HUGE increase in the number of molecules, 
    as a single SMILES may lead to MANY different ionization states.
    
    The larger the pH range, the larger the number of allowed states.
    *****************************************************************
    
    Given a dataframe with a 'ROMol' column, used Dimorphite-DL[*] 
    to enumerate all possible ionization states in a pH range. Allowed arguments are:
    'min_ph' : Default = 6.4,
    'max_ph' : Default = 8.4,
    'pka_precision' : Default = 1.0
    
    [*] Dimorphite-DL:
    Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
    open-source program for enumerating the ionization states of drug-like small
    molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.
    
    """    
    from vsdb.dimorphite_dl import dimorphite_dl

    ionization_states = 0
    initial_size = len(df)
    new_df = pd.DataFrame(columns=df.columns)
    
    print("Enumerating ionization states with Dimorphite-DL")
    print("Initial size = ", initial_size)
    for index in tqdm(df.index, desc="Enumerating"): 
        smi = df.loc[index]['SMILES']

        protonated = list(dimorphite_dl.Protonate({'smiles':smi, 
                                                   'min_ph':min_ph, 
                                                   'max_ph':max_ph, 
                                                   'pka_precision':pka_precision,
                                                   'test':False, 
                                                   'return_as_list':True, 
                                                   'label_states':False}))
        new_smiles = len(protonated)
        
        for smi in protonated:
            new_rec = df.loc[index].copy()
            new_rec['SMILES']   = smi
            new_rec['ROMol']    = Chem.MolFromSmiles(smi)
            new_rec['InChI']    = Chem.inchi.MolToInchi(new_rec['ROMol'])
            new_rec['InChIKey'] = Chem.inchi.MolToInchiKey(new_rec['ROMol'])
            new_df = new_df.append(new_rec,ignore_index=True)
            
    final_size = len(new_df)
    print("Final size  = ", final_size)
    print(f"Added {final_size - initial_size} new molecules to the DataFrame.")
    
    return new_df.reset_index(drop=True)

