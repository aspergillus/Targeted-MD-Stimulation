import sys
import math
import time
import pymol
import Bio.PDB
import warnings
import numpy as np
import pandas as pd
from pymol import cmd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import three_to_one

warnings.filterwarnings("ignore")

# Start the timer
start_time = time.time()

# Define a class to mutate a specific residue 
class MutateResidue(Select):
    def __init__(self, chain_id, res_id, new_res_name):
        self.chain_id = chain_id
        self.res_id = res_id
        self.new_res_name = new_res_name

    def accept_residue(self, residue):
        if residue.id[1] == self.res_id and residue.get_parent().id == self.chain_id:
            residue.resname = self.new_res_name
        return True

def mutate(file_name, chain_name, atom_number, mutation):

    # Parse the PDB file using BioPytohon
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("protein", file_name)
    mutator = MutateResidue(chain_name, atom_number, mutation)  # Apply the mutation
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(f"Modified_PDB_Files/{file_name.split('/')[1].split('.')[0]}_mutated.pdb", select=mutator)
    print("Mutation successfully")

def load_and_center_protein(pdb_file, padding=10.0):
    # Clear all objects to avoid duplication
    cmd.delete("all")

    # Load the protein structure
    cmd.load(pdb_file, "protein")

    # Remove water molecules
    cmd.remove("resn HOH")

    # Calculate the bounding box of the protein
    min_corner, max_corner = cmd.get_extent("protein")
    min_corner = np.array(min_corner)
    max_corner = np.array(max_corner)
    protein_center = (min_corner + max_corner) / 2.0

    # Calculate grid limits (add padding)
    min_corner_with_padding = min_corner - padding
    max_corner_with_padding = max_corner + padding
    grid_center = (min_corner_with_padding + max_corner_with_padding) / 2.0

    # Translate protein to the center of the grid
    translation_vector = grid_center - protein_center
    cmd.translate(list(translation_vector), "protein")

    print("Protein centered successfully!")
    return grid_center, min_corner_with_padding, max_corner_with_padding

def create_grids(pdb_file, min_corner, max_corner, grid_spacing=5.0):
    # Calculate the number of grids required along each axis
    protein_size = max_corner - min_corner
    num_grids = np.ceil(protein_size / grid_spacing).astype(int)

    print(f"Number of grids: {num_grids}")

    subgrid_spacing = (max_corner - min_corner) / num_grids
    grid_labels = "abcdefghijklmnopqrstuvwxyz"  # For labeling rows and columns
    grid_residues = {}  # Dictionary to store residues for each grid label

    for i in range(num_grids[0]):
        for j in range(num_grids[1]):
            for k in range(num_grids[2]):
                # Define the corners of each subgrid
                subgrid_min_corner = min_corner + np.array([i, j, k]) * subgrid_spacing
                subgrid_max_corner = subgrid_min_corner + subgrid_spacing

                # Compute the center of the subgrid for labeling
                subgrid_center = (subgrid_min_corner + subgrid_max_corner) / 2.0
                label_text = f"{grid_labels[i]}{j + 1}{k + 1}"

                # Check if the protein intersects with the subgrid
                selection_name = f"subgrid_sel_{i}_{j}_{k}"
                cmd.select(
                    selection_name,
                    f"protein within {grid_spacing} of (x > {subgrid_min_corner[0]} and x < {subgrid_max_corner[0]} "
                    f"and y > {subgrid_min_corner[1]} and y < {subgrid_max_corner[1]} "
                    f"and z > {subgrid_min_corner[2]} and z < {subgrid_max_corner[2]})"
                )

                # Collect residues if the subgrid intersects with the protein
                if cmd.count_atoms(selection_name) > 0:
                    residues = cmd.get_model(selection_name).atom
                    residue_list = list(set(f"{atom.resn}{atom.resi}" for atom in residues))
                    grid_residues[label_text] = residue_list

                    # Create one sphere at the center of the subgrid
                    cmd.pseudoatom(f"subgrid_{label_text}", pos=subgrid_center.tolist())

                    # Add label at the center of the subgrid
                    cmd.label(f"subgrid_{label_text}", f'"{label_text}"')  # Assign the intended text to the label
                    cmd.set("label_size", -0.5)  # Adjust label size
                    cmd.set("label_color", "white")  # Set label color

    # Save grid residues to a file
    proteinName = pdb_file.split('/')[1][:-4]
    with open(f"gridOutput/{proteinName}_grid_residues.txt", "w") as f:
        for grid, residues in grid_residues.items():
            # f.write(f"{grid}: {', '.join(residues)}\n") # To write residues falling in the grid and represented in comman-separated string 
            f.write(f"{grid}: {len(residues)}\n")

    # Group and display the grids
    cmd.group("filtered_grids", "subgrid_*")
    cmd.show("spheres", "filtered_grids")
    cmd.set("sphere_scale", 0.3, "filtered_grids")  # Adjust pseudoatom size
    # cmd.save(f"Modified_PDB_Files/{proteinName}_grid.pse")
    print("Filtered grids created, labeled, and residues saved to 'grid_residues.txt' successfully!")

def unfold_protein(input_file, selected_chain, residueID, mutation):

    proteinName = input_file.split('/')[1][:-4]
    # Parse the PDB file using BioPytohon
    parser = PDBParser()
    structure = parser.get_structure("protein", input_file)

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

    # Iterate through residues
    df = pd.DataFrame(columns=("Residue_ID", "Residue_Name"))
    for model in structure:
        for chain in model:
            if chain.id == selected_chain:
                for i, residue in enumerate(chain):
                    residue_id = residue.id[1]
                    if residue_id in range((residueID - 10) - 1, (residueID + 10) + 2):
                        residue_name = residue.resname
                        df.loc[len(df)] = residue_id, residue_name

    # Pymol reinitialize
    # cmd.reinitialize()
    # cmd.load(f"Modified_PDB_Files/{proteinName}_grid.pse")

    # Make Tuples of the Region to unfold on the chain
    region_to_unfold_tuples = []
    region_to_unfold_selection_tuples = []
    for index, row in df.iterrows():
        region_to_unfold_selection_tuples.append((f"{row.iloc[1]}_{row.iloc[0]}",f"resn {row.iloc[1]} and resi {row.iloc[0]} and chain {selected_chain}"))
        region_to_unfold_tuples.append((f"{row.iloc[1]}`{row.iloc[0]}",f"resn {row.iloc[1]} and resi {row.iloc[0]} and chain {selected_chain}"))
    region_to_unfold_tuples = tuple(region_to_unfold_tuples)
    region_to_unfold_selection_tuples = tuple(region_to_unfold_selection_tuples)

    proteinObjectName = input_file.split('/')[1][:-4]
    previousResidue, currentResidue, nextResidue = (None for i in range(3))

    # Unfolding the protein to linearize
    cmd.set_name("protein", proteinName)
    for i in range(1, len(region_to_unfold_tuples) - 1):
        previousResidue = region_to_unfold_tuples[i-1][0]
        currentResidue = region_to_unfold_tuples[i][0]
        nextResidue = region_to_unfold_tuples[i+1][0]

        # Phi Angle
        cmd.set_dihedral(f"/{proteinObjectName}//{selected_chain}/{previousResidue}/C", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/N", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/CA", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/C", -120, state=1, quiet=1)

        # Psi Angle
        cmd.set_dihedral(f"/{proteinObjectName}//{selected_chain}/{currentResidue}/N", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/CA", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/C", 
                         f"/{proteinObjectName}//{selected_chain}/{nextResidue}/N", 140, state=1, quiet=1)

        # Omega Angle
        cmd.set_dihedral(f"/{proteinObjectName}//{selected_chain}/{currentResidue}/CA", 
                         f"/{proteinObjectName}//{selected_chain}/{currentResidue}/C", 
                         f"/{proteinObjectName}//{selected_chain}/{nextResidue}/N", 
                         f"/{proteinObjectName}//{selected_chain}/{nextResidue}/CA", 180, state=1, quiet=1)

    # cmd.rebuild()
    cmd.save(f"Modified_PDB_Files/{proteinName}_mutated_grid_unfolded_352.pse")

def main(pdb_file, grid_spacing=5.0):
    # selected_chain, mutation, residueID = "A", "PHE", "100" # Inputs variables
    selected_chain, mutation, residueID = "A", "VAL", "252" # Inputs variables
    # selected_chain, mutation, residueID = "A", "ILE", "352" # Inputs variables
    mutate(pdb_file, selected_chain, residueID, mutation)
    grid_center, min_corner, max_corner = load_and_center_protein(pdb_file)
    create_grids(pdb_file, min_corner, max_corner, grid_spacing)
    unfold_protein(pdb_file, selected_chain, int(residueID), mutation)

# Example usage
pdb_path = "PDB_Structure/4fdi.pdb"  # Replace with your PDB file path
main(pdb_path)
print(f"Process time: {time.time() - start_time:.2f} seconds")
