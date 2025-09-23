"""
Creates VTK files from OpenFOAM case directories.
Uses subsets JSON (see ./subsets)to divide into train, valid, test.
"""


import os
from utils import run_foam
import shutil
import json
import argparse

parser = argparse.ArgumentParser(description='Process dataset subsets')
parser.add_argument('--machine', type=str, default='mac', help='Device type')
args = parser.parse_args()

machine = args.machine

if machine == 'mac':
    data_dpath = '/Users/adamdray/Code/data'
    on_mac = True
elif machine == 'csd3':
    data_dpath = '/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246'
    on_mac = False

## === INPUTS ---

simulation_name = 'ellipseflow3'
use_subsets = True
time = None

## ---

sim_dir = f"{data_dpath}/openfoam/{simulation_name}"
vtk_output_dir = f"{data_dpath}/vtk/{simulation_name}"

if use_subsets:
    # Load dataset subset definitions from JSON
    with open('generate/subsets/ellipseflow3.json', 'r') as f:
        subsets = json.load(f)
else:
    meshes = os.listdir(sim_dir)
    subsets = {
        'train' : [],
        'valid' : meshes,
        'test' : []
    }

# Process each subset (train, valid, test) separately
for subset, meshes in subsets.items():
    print(f"\nProcessing {subset} subset...")

    # Create subset directory
    subset_dir = os.path.join(vtk_output_dir, subset)
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    else:
        shutil.rmtree(subset_dir)
        os.makedirs(subset_dir)

    # Process each mesh in this subset
    for mesh in meshes:
        case_dir = os.path.join(sim_dir, mesh)

        if not os.path.isdir(case_dir):
            print(f"Warning: {mesh} directory not found in {sim_dir}")
            continue

        print(f"Converting simulation results in {case_dir} to VTK format")
        if time:
            run_foam(f"foamToVTK -surfaceFields -time 0:{time}", case_dir, on_mac)
        else:
            run_foam("foamToVTK -surfaceFields -time $(foamListTimes -withZero | awk 'NR % 2 == 0' | tr '\n' ',')", case_dir, on_mac) #-time $(foamListTimes -noZero | awk 'NR % 2 == 0' | tr '\n' ',')
        print(f"Converted results for {case_dir} to VTK format")

        # Move the VTK files to the appropriate subset directory
        vtk_source = os.path.join(case_dir, "VTK")
        vtk_target = os.path.join(vtk_output_dir, subset, mesh)

        if os.path.exists(vtk_source):
            shutil.move(vtk_source, vtk_target)
            print(f"Moved VTK files for {mesh} to {subset} set")

        # Copy meta.json if it exists
        meta_source = os.path.join(case_dir, "meta.json")
        meta_target = os.path.join(vtk_output_dir, subset, mesh, "meta.json")

        if os.path.exists(meta_source):
            shutil.copy(meta_source, meta_target)

            # Edit meta.json to multiply physics.dt by 2
            with open(meta_target, 'r') as f:
                meta_data = json.load(f)
                meta_data['physics']['dt'] *= 2
                with open(meta_target, 'w') as f:
                    json.dump(meta_data, f, indent=2)
            print(f"Copied meta.json for {mesh}")

print("\nAll subsets processed successfully!")
