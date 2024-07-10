from Bio.PDB import PDBParser, PDBIO
import warnings
import Bio.PDB
import math
import pandas as pd
import pymol
from pymol import cmd
warnings.filterwarnings("ignore")

# Specify file paths
input_pdb_file = "4fdi.pdb"
output_pdb_file = "modified_protein.pdb"
selected_chain = "A"
mutation_residue_number = 250

# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure("protein", input_pdb_file)
    
# Function to convert three-letter codes to one-letter codes
three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def convert_to_one_letter(three_letter_code):
    three_letter_code = three_letter_code
    if three_letter_code in three_to_one:
        return three_to_one[three_letter_code]
    else:
        return None
    
# Iterate through residues 40 to 49 (inclusive)
df = pd.DataFrame(columns=("Residue_ID", "Residue_Name"))
for model in structure:
    for chain in model:
        if chain.id == selected_chain:
            for i, residue in enumerate(chain):
                residue_id = residue.id[1]
                if residue_id in range((mutation_residue_number - 10), (mutation_residue_number + 10)):
                    residue_name = residue.resname
                    df.loc[len(df)] = residue_id, residue_name
# print(df)
cmd.reinitialize()
cmd.load(input_pdb_file)

# Make Selection object on the specify chain 
for index, row in df.iterrows():
    cmd.create(f"{row[1]}_{row[0]}", f"resn {row[1]} and resi {row[0]} and chain A")
    
object_list = cmd.get_object_list()[1:]
resLst = df.Residue_ID

preObjVal, currObjVal, nextobjVal, nextTonextobjVal, preResVal, currResVal, nextResVal, nextTonextResVal = (None for i in range(8))
for i in range(1, len(object_list), 4):
    preObjVal, currObjVal, nextobjVal, nextTonextobjVal = object_list[i - 1:i + 3]
    preResVal, currResVal, nextResVal, nextTonextResVal = resLst[i - 1:i + 3]
    
    # print(preResVal, currResVal, nextResVal, nextTonextResVal)

    preObjVal = preObjVal.replace("_","`")
    currObjVal = currObjVal.replace("_","`")
    nextobjVal = nextobjVal.replace("_","`")
    nextTonextobjVal = nextTonextobjVal.replace("_","`")
    
    cmd.set_dihedral(f"/4fdi//A/{preObjVal}/N", f"/4fdi//A/{preObjVal}/CA", f"/4fdi//A/{preObjVal}/C", f"/4fdi//A/{currObjVal}/N", -180, quiet=0) # phi
    cmd.set_dihedral(f"/4fdi//A/{currObjVal}/CA", f"/4fdi//A/{currObjVal}/C", f"/4fdi//A/{nextobjVal}/N", f"/4fdi//A/{nextobjVal}/CA", 180, quiet=0) # psi
    cmd.set_dihedral(f"/4fdi//A/{nextobjVal}/C", f"/4fdi//A/{nextTonextobjVal}/N", f"/4fdi//A/{nextTonextobjVal}/CA", f"/4fdi//A/{nextTonextobjVal}/C", 180, quiet=0) # omega
    
# cmd.rebuild()
cmd.save(output_pdb_file, state=-1, format='pdb')