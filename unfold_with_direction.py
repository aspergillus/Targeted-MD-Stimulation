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

warnings.filterwarnings("ignore")

# Start the timer
start_time = time.time()

class MutateResidue(Select):
    def __init__(self, chain_id, res_id, new_res_name):
        self.chain_id = chain_id
        self.res_id = res_id
        self.new_res_name = new_res_name

    def accept_residue(self, residue):
        if residue.id[1] == self.res_id and residue.get_parent().id == self.chain_id:
            residue.resname = self.new_res_name
        return True

def mutate(file_name, chain_name, res_id, mutation):
    """Mutate a specific residue in the protein structure"""
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("protein", file_name)
    mutator = MutateResidue(chain_name, res_id, mutation)
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(f"Modified_PDB_Files/{file_name.split('/')[1].split('.')[0]}_mutated.pdb", select=mutator)
    print("Mutation successful!")

def load_and_center_protein(pdb_file, padding=10.0):
    """Center protein in a grid container"""
    cmd.delete("all")
    cmd.load(pdb_file, "protein")
    cmd.remove("resn HOH")
    
    min_corner, max_corner = cmd.get_extent("protein")
    min_corner = np.array(min_corner)
    max_corner = np.array(max_corner)
    protein_center = (min_corner + max_corner) / 2.0
    
    grid_center = (min_corner - padding + max_corner + padding) / 2.0
    translation_vector = grid_center - protein_center
    cmd.translate(list(translation_vector), "protein")
    
    return min_corner - padding, max_corner + padding

def create_grids(pdb_file, min_corner, max_corner, grid_spacing=5.0):
    """Create 3D grid system around protein"""
    protein_size = max_corner - min_corner
    num_grids = np.ceil(protein_size / grid_spacing).astype(int)
    subgrid_spacing = (max_corner - min_corner) / num_grids
    
    grid_residues = {}
    for i in range(num_grids[0]):
        for j in range(num_grids[1]):
            for k in range(num_grids[2]):
                subgrid_min = min_corner + np.array([i, j, k]) * subgrid_spacing
                subgrid_max = subgrid_min + subgrid_spacing
                
                sel_name = f"subgrid_{i}_{j}_{k}"
                cmd.select(sel_name, f"protein within {grid_spacing} of (x>{subgrid_min[0]} and x<{subgrid_max[0]} and y >{subgrid_min[1]} and y<{subgrid_max[1]} and z>{subgrid_min[2]} and z<{subgrid_max[2]})")
                
                if cmd.count_atoms(sel_name) > 0:
                    atoms = cmd.get_model(sel_name).atom
                    residues = list({f"{atom.chain}:{atom.resn}{atom.resi}" for atom in atoms})
                    grid_residues[f"{i}{j}{k}"] = residues

    protein_name = pdb_file.split('/')[1][:-4]
    with open(f"gridOutput/{protein_name}_grid_residues.txt", "w") as f:
        for grid, res in grid_residues.items():
            f.write(f"{grid}: {len(res)}\n")
    
    return grid_residues

def analyze_unfolding_direction(selection):
    """Determine optimal unfolding direction using spatial distribution"""
    coords = []
    cmd.iterate_state(1, selection, "coords.append((x,y,z))", space={'coords': coords})
    coords = np.array(coords)
    
    # Calculate principal axes using PCA
    cov_matrix = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Determine direction sign based on median position
    median_position = np.median(coords, axis=0)
    direction_sign = '+' if np.dot(main_axis, median_position) > 0 else '-'
    
    axis_names = ['x', 'y', 'z']
    return f"{axis_names[np.argmax(np.abs(main_axis))]}{direction_sign}"

def create_anchors(selection, buffer=10):
    """Create stabilization anchors around target region"""
    anchor_sel = f"byres ({selection}) expand {buffer}"
    cmd.select("anchors", anchor_sel)
    cmd.disable("anchors")
    print(f"Created {cmd.count_atoms('anchors')} stabilization anchors")

def smart_linearize(selection, direction, phi=-120, psi=140, omega=180):
        
    """Direction-aware unfolding with stabilization"""
    # Store original coordinates for reference
    original_state = cmd.get_state()
    
    # Convert residue IDs to integers and sort
    residues = sorted([int(a.resi) for a in cmd.get_model(selection).atom])
    residues = list(dict.fromkeys(residues))  # Remove duplicates while preserving order
    
    chain = cmd.get_model(selection).atom[0].chain
    
    for i, resi in enumerate(residues[:-1]):
        next_resi = resi + 1  # Now works with integers
        
        # cmd.select(f"{chain}_{resi}-{next_resi}", f"chain {chain} and resi {resi}-{next_resi}")
        
        cmd.set_dihedral(f"/protein//{chain}/{resi}/C", 
                         f"/protein//{chain}/{next_resi}/N", 
                         f"/protein//{chain}/{next_resi}/CA", 
                         f"/protein//{chain}/{next_resi}/C", phi + i*10, quiet=0)  # Small stepwise changes
                         
        cmd.set_dihedral(f"/protein//{chain}/{resi}/CA", 
                         f"/protein//{chain}/{resi}/C", 
                         f"/protein//{chain}/{next_resi}/N", 
                         f"/protein//{chain}/{next_resi}/CA", psi + i*10, quiet=0)
                         
        cmd.set_dihedral(f"/protein//{chain}/{resi}/N", 
                         f"/protein//{chain}/{resi}/CA", 
                         f"/protein//{chain}/{resi}/C", 
                         f"/protein//{chain}/{next_resi}/N", omega, quiet=0)
        
        
        # cmd.sculpt_activate(f"{chain}_{resi}-{next_resi}")  # Activate sculpting mode for selection
        # cmd.sculpt_iterate(f"{chain}_{resi}-{next_resi}", cycles=50)  # Perform 50 iterations
        # cmd.sculpt_deactivate(f"{chain}_{resi}-{next_resi}")  # Deactivate sculpting when done
        
        cmd.rebuild()

def unfold_protein(input_file, chain, res_id_range):
    """Main unfolding workflow"""
    # Setup
    selection = f"chain {chain} and resi {res_id_range[0]}-{res_id_range[1]}"
    cmd.load(input_file, "target")
    
    # Grid analysis
    direction = analyze_unfolding_direction(selection)
    print(f"Optimal unfolding direction: {direction}")
    
    # Stabilization
    create_anchors(selection)
    
    # Direction-aware unfolding
    print(selection)
    smart_linearize(selection, direction)
    
    # Final cleanup
    cmd.save(f"Modified_PDB_Files/{input_file.split('/')[-1][:-4]}_unfolded.pdb")

def main(pdb_file):
    """Execution workflow"""
    # Example: Unfold residues 340-360 in chain A
    mutate(pdb_file, "A", 350, "ALA")  # Optional mutation
    min_corner, max_corner = load_and_center_protein(pdb_file)
    grid_data = create_grids(pdb_file, min_corner, max_corner)
    unfold_protein(pdb_file, "A", (340, 360))

if __name__ == "__main__":
    pdb_path = "PDB_Structure/4fdi.pdb"
    main(pdb_path)
    print(f"Process completed in {time.time()-start_time:.2f} seconds")