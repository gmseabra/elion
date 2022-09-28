import pandas as pd
from vsdb import preprocess
from chembl_webresource_client.new_client import new_client

def retrieve_similars(smiles_str, min_similarity=80, precision=3):
    """
    Receives a SMILES string, and returns the structures from ChEMBL with at
    least <min_similarity> tanimoto similarity to the compound.

    PARAMETERS:
        - smiles_str    : The SMILES string of the query compound. (Required)
        - min_similarity: The minimum similarity to consider. Default: 80% 

    Returns a list of dictionaries. Each element in the list is a dictionary
    with the fields:
        - Name:       The ChEMBL ID of the molecule
        - SMILES:     The SMILES string
        - Similarity: The numerical Tanimoto similarity to the query compound.

    """
    assert 40 <= min_similarity <= 100, \
          f"Invalid Similarity Score supplied: {min_similarity}. It should be between 40 and 100"

    similarity = new_client.similarity
    _ = similarity.filter(smiles=smiles_str, similarity=min_similarity)\
                  .only(["molecule_chembl_id","molecule_structures","similarity"])

    # Curates. All we need is the SMILES string.

    res = []
    for molecule in _:
        this_mol = {}
        this_mol["Name"]   = molecule["molecule_chembl_id"]
        this_mol['SMILES'] = molecule["molecule_structures"]["canonical_smiles"]
        this_mol['Similarity'] = round(float(molecule['similarity']),precision)
        res.append(this_mol)

    return res

def get_df_similars(parents_df, smiles_col="SMILES", name_col="Name", min_similarity=80):
    """
    Receives a DataFrame with molecules in SMILES format, and returns a new
    DataFrame with similar molecules resulting from a query to ChEMBL.
    
    The new DataFrame is pre-processed to:
    1. Remove molecules that are identical to the parent
    2. Remove repetitions (same similarity to the parent)
    3. Preprocess and standardize the molecules
    """
    new_mols = []
    for ind, parent in parents_df.iterrows():
        parent_smi   = parent[smiles_col]
        new_similars = retrieve_similars(parent_smi, min_similarity)

        for similar in new_similars:
            similar["Parent"] = parent[name_col]

        new_mols.extend(new_similars)
    
    new_mols = pd.DataFrame(new_mols)

    new_mols = new_mols[ new_mols['Similarity'] < 100 ]
    new_mols = new_mols.drop_duplicates(subset=['Similarity','Parent'], ignore_index=True)
    new_mols = preprocess.prepare_data_pipeline(new_mols)

    return new_mols


if __name__ == "__main__":

    """
    This is for testing purposes.
    """    
    smiles_str = "CO[C@@H](CCC#C\C=C/CCCC(C)CCCCC=C)C(=O)[O-]"
    res = retrieve_similars(smiles_str, min_similarity=40)
    
    for molecule in res:
        for prop_key in molecule:
            print(f"{prop_key:20s} :  {molecule[prop_key]}")
        print("="*40)
   
    print(len(res))
