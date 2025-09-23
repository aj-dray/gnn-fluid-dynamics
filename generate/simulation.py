'''
Run OpenFOAM simulations using case directories in ./openfoam
Config is hardcoded in the # === INPUTS === section below
'''

## === LIBRARIES ---

import subprocess
import utils
import os
import numpy as np
import time
import json
import argparse
import shutil

## === MODULES ---

from utils import modify_foamDict, run_foam

## === FUNCTIONS ---

def clean_results_preserve_mesh(case_dir):
    """Clean OpenFOAM results but preserve mesh"""
    # Remove time directories except 0/
    subprocess.run(f"cd {case_dir} && rm -rf [1-9]* [1-9]*.[0-9]* log.*", shell=True)

    # Clean postProcessing but keep constant/ and system/
    subprocess.run(f"cd {case_dir} && rm -rf postProcessing", shell=True)

    print(f"Cleaned results in {case_dir} while preserving mesh")


def modify_openfoam_case(
    case_dir: str,
    U: float,
    wall_type: str,
    nu: float,
    dt: float,
    end_time: float,
    write_interval: int,
    noise_amp: float = 0.0,
    noise_scale: float = 0.1,
    on_mac: bool = False,
):

    tp = os.path.join(case_dir, "constant", "transportProperties")
    modify_foamDict(tp, {"nu": f"nu [0 2 -1 0 0 0 0] {nu}"}, on_mac=on_mac)

    cd = os.path.join(case_dir, "system", "controlDict")
    entries = {
        "deltaT": f"{dt}",
        "startTime": f"{0}",
        "endTime": f"{end_time}",
        "writeInterval": f"{write_interval}"
    }
    modify_foamDict(cd, entries, on_mac=on_mac)

    # tp2 = os.path.join(case_dir, "constant", "turbulenceProperties")
    # modify_foamDict(tp2, {"simulationType": simulation_type})

    Ufile = os.path.join(case_dir, "0", "U")
    U_uniform = f"uniform ({U} 0 0)"
    if noise_amp <= 0.0:
        # pure fixedValue BC
        entries = {
            "internalField": U_uniform,
            "boundaryField.inlet.type": "fixedValue",
            "boundaryField.inlet.value": U_uniform,
            "boundaryField.walls.type": wall_type,
        }
    else:
        # random‐noise BC
        entries = {
            "internalField": U_uniform,
            "boundaryField.inlet.type": "randomUniformFixedValue",
            "boundaryField.inlet.meanValue": U_uniform,
            "boundaryField.inlet.amplitude": f"{noise_amp}",
            "boundaryField.inlet.scale": f"{noise_scale}"
        }

    modify_foamDict(Ufile, entries, on_mac=on_mac)

    print(f"Modifed opeanFoam files in {case_dir}:")


def edit_blockMesh_res(res, target_dir):
    # Edit blockMeshDict to set the mesh resolution
    blockMeshDict_path = os.path.join(target_dir, "system", "blockMeshDict")
    with open(blockMeshDict_path, "r") as f:
        blockmesh_lines = f.readlines()
    new_lines = []
    for line in blockmesh_lines:
        if "hex (0 1 2 3 4 5 6 7)" in line:
            # Replace the resolution tuple (e.g., (64 64 1)) with the current res
            new_line = f"    hex (0 1 2 3 4 5 6 7) ({res} {res} 1) simpleGrading (1 1 1)\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    with open(blockMeshDict_path, "w") as f:
        f.writelines(new_lines)

## === MAIN ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run OpenFOAM simulations')
    parser.add_argument('--machine', type=str, help='Device being used to determine datapath')
    parser.add_argument('--array_id', type=int, default=-1, help='Slurm array ID for parallel processing')
    parser.add_argument('--array_total', type=int, default=1, help='Total number of Slurm array jobs')
    args = parser.parse_args()

    # dataset_name = "laminar_ellipse_mini"
    # mesh_fpath = "/Users/adamdray/Code/data/gmsh/single_circle/extruded" #"/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246/custom/gmsh/of_test1/"
    # generation_dir = f"/Users/adamdray/Code/project/generate/openfoam/{dataset_name}" #/home/ajd246/code/project-dev/generate/openfoam/simulation1"
    # base_target_dir_root = f"/Users/adamdray/Code/data/openfoam/{dataset_name}" # Root directory for storing simulation results

    # if os.path.exists(generation_dir):
    #     run_foam("./Allclean", generation_dir)
    #     print("Run Allclean in the generation directory.")
    # else:
    #     # If generation_dir doesn't exist, create it based on a template or raise error
    #     # Assuming a template case exists or needs to be copied here.
    #     # For now, raising an error if it doesn't exist.
    #     raise FileNotFoundError(f"Generation directory {generation_dir} does not exist. Please ensure it contains a base OpenFOAM case.")


    if args.machine == 'mac':
        gen_base = '/Users/adamdray/Code/MPhil_Project/project'
        data_base = '/Users/adamdray/Code/data'
        on_mac = True
    elif args.machine == 'csd3':
        gen_base = '/home/ajd246/code/project'
        data_base = '/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246'
        on_mac = False

    total_start = time.time()

    ## === INPUTS ---
    # Calculate input velocity
    # nu = 1e-3
    # Re_min = 20
    # Re_max = 40
    # num_U = 1
    # num_timesteps = 500
    # CFL = 0.40
    # write_interval = 10

    # resolutions = [64, 128, 256, 512]

    # simulation_type = "taylor_green" # choose generation case directory
    # simulation_name = "taylor_green-big" # save directory

    ## === INPUT LAMINAR ---

    simulation_type = "laminar_ellipse"
    simulation_name = "ellipseflow5"
    dataset_name = "ellipseflow6" # mesh directory
    subset = "dataset_single"
    end_timesteps = 14000 # set off input variables
    log_freq = 10
    CFL = 0.5

    ## ---

    if simulation_type == "laminar_ellipse":

        mesh_fpath = f"{data_base}/gmsh/{dataset_name}"
        generation_dir = f"{gen_base}/generate/openfoam/{simulation_type}"
        base_target_dir_root = f"{data_base}/openfoam/{simulation_name}" # Root directory for storing simulation results

        # Check if generation directory exists
        if not os.path.exists(generation_dir):
            raise FileNotFoundError(f"Generation directory {generation_dir} does not exist. Please ensure it contains a base OpenFOAM case.")

        # Get list of mesh folders in mesh_fpath
        with open(f'generate/subsets/{subset}.json', 'r') as f:
            subsets = json.load(f)
        mesh_names = []
        for subset, meshes in subsets.items():
            mesh_names.extend(meshes)

        # split between array jobs
        if args.array_id >= 0:
            # Split the meshes among the array jobs
            meshes_per_job = int(np.ceil(len(mesh_names) / args.array_total))
            start_idx = args.array_id * meshes_per_job
            end_idx = min(start_idx + meshes_per_job, len(mesh_names))
            mesh_names = mesh_names[start_idx:end_idx]
            print(f"Array job {args.array_id}/{args.array_total} processing {len(mesh_names)} meshes: {start_idx} to {end_idx-1}")

        print(f"Processing {len(mesh_names)} meshes in {mesh_fpath}...")

        # loop through the mesh files
        for i, mesh in enumerate(mesh_names):
            # Manage directories
            meshpath = os.path.join(mesh_fpath, mesh)
            sim_name = mesh
            target_dir = os.path.join(base_target_dir_root, sim_name)
            if os.path.exists(target_dir):
                print(f"Removing existing target directory: {target_dir}")
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

            # First copy the base OpenFOAM case to the target directory
            subprocess.run(["cp", "-r", os.path.join(generation_dir, "."), target_dir])
            print(f"Copied base case to {target_dir}")

            # Clean the target directory
            run_foam("./Allclean", target_dir, on_mac=on_mac)

            # Generate the mesh directly in the target directory
            print(f"Generating OpenFOAM mesh for {mesh} in target directory")
            utils.gmsh_to_foam(meshpath, target_dir, on_mac=on_mac)

            # Read in meta.json file
            metapath = os.path.join(mesh_fpath, mesh, "meta.json")
            with open(metapath, 'r') as f:
                meta_dict = json.load(f)

            # Modify the case files directly in the target directory
            u_in = meta_dict["boundary_conditions"]["inlet"]["value"]
            nu = meta_dict["physics"]["nu"]
            dt = meta_dict["physics"]["dt"] * CFL

            # Calculate end time based on timesteps
            end_time = end_timesteps * dt

            # Only update the dt in the meta dictionary
            meta_dict["physics"]["dt"] = dt * log_freq

            # Write the modified meta.json to the target directory
            with open(os.path.join(target_dir, "meta.json"), "w") as f:
                json.dump(meta_dict, f, indent=2)

            print("dt:", dt)
            print("end_time:", end_time)

            modify_openfoam_case(
                case_dir=target_dir,
                U=u_in,
                wall_type=meta_dict["boundary_conditions"]["walls"]["type"],
                nu=nu,
                dt=dt,
                end_time=end_time,
                write_interval=log_freq,
                noise_amp=0.0,
                noise_scale=0.0,
                on_mac=on_mac,
            )

            # Run simulation in the target directory
            print(f"Running pimpleFoam in {target_dir}")
            start = time.time()
            run_foam("pimpleFoam | tee pimpleFoam.log", target_dir, on_mac=on_mac)
            end = time.time()
            print(f"Finished simulation for {sim_name} in {end - start:.2f} seconds")
            with open(os.path.join(target_dir, "time.log"), "w") as f:
                f.write(f"{end - start:.2f}\n")

            # -------------------------------------------
            # # INLINE CONVERSION
            # # -------------------------------------------
            # # Map mesh id to subset
            # for sname, meshes in subsets.items():
            #     if mesh in meshes:
            #         subset = sname
            #         break
            # else:
            #     subset = "unknown"
            # sim_dir = f"{data_base}/openfoam/{simulation_name}"
            # vtk_output_dir = f"{data_base}/vtk/{simulation_name}"

            # case_dir = os.path.join(sim_dir, mesh)

            # if not os.path.isdir(case_dir):
            #     print(f"Warning: {mesh} directory not found in {sim_dir}")
            #     continue

            # print(f"Converting simulation results in {case_dir} to VTK format")
            # run_foam("foamToVTK -surfaceFields -time $(foamListTimes -withZero | awk 'NR % 2 == 0' | tr '\n' ',')", case_dir, on_mac) #-time $(foamListTimes -noZero | awk 'NR % 2 == 0' | tr '\n' ',')
            # print(f"Converted results for {case_dir} to VTK format")

            # # Move the VTK files to the appropriate subset directory
            # vtk_source = os.path.join(case_dir, "VTK")
            # vtk_target = os.path.join(vtk_output_dir, subset, mesh)

            # if os.path.exists(vtk_source):
            #     shutil.move(vtk_source, vtk_target)
            #     print(f"Moved VTK files for {mesh} to {subset} set")

            # # Copy meta.json if it exists
            # meta_source = os.path.join(case_dir, "meta.json")
            # meta_target = os.path.join(vtk_output_dir, subset, mesh, "meta.json")

            # if os.path.exists(meta_source):
            #     shutil.copy(meta_source, meta_target)

            #     # Edit meta.json to multiply physics.dt by 2
            #     with open(meta_target, 'r') as f:
            #         meta_data = json.load(f)
            #         meta_data['physics']['dt'] *= 2
            #         with open(meta_target, 'w') as f:
            #             json.dump(meta_data, f, indent=2)
            #     print(f"Copied meta.json for {mesh}")

            # # Delete the case_dir after processing
            # if os.path.exists(case_dir):
            #     shutil.rmtree(case_dir)
            #     print(f"Deleted case directory {case_dir}")

    elif simulation_type == 'taylor_green':

        mesh_fpath = f"{data_base}/gmsh/{dataset_name}"
        generation_dir = f"{gen_base}/generate/openfoam/{simulation_type}"
        base_target_dir_root = f"{data_base}/openfoam/{simulation_name}"

        for res in resolutions:
            mesh = f"{res}x{res}"

            meshpath = os.path.join(mesh_fpath, mesh)
            target_dir = os.path.join(base_target_dir_root, mesh)
            if os.path.exists(target_dir):
                subprocess.run(["rm", "-rf", target_dir])
            os.makedirs(target_dir)

            # Copy the base OpenFOAM case to the target directory
            subprocess.run(["cp", "-r", os.path.join(generation_dir, '.'), target_dir])
            print(f"Copied base case to {target_dir}")

            # Clean the target directory
            run_foam("./Allclean", target_dir, on_mac=on_mac)

            # Generate the mesh directly in the target directory
            edit_blockMesh_res(res, target_dir)
            print(f"Generating OpenFOAM mesh for {mesh} in target directory")
            run_foam("blockMesh | tee blockMesh.log", target_dir, on_mac=on_mac)

            # Run simulation in the target directory
            print(f"Running pimpleFoam in {target_dir}")
            start = time.time()
            run_foam("pimpleFoam | tee pimpleFoam.log", target_dir, on_mac=on_mac)
            end = time.time()
            print(f"Finished simulation for {mesh} in {end - start:.2f} seconds")
            with open(os.path.join(target_dir, "time.log"), "w") as f:
                f.write(f"{end - start:.2f}\n")


            # total_end = time.time()
            # # Create a job-specific time log in the base directory
            # job_time_log = f"time_process_{args.array_id }.log" if args.array_id >= 0 else "time.log"
            # with open(os.path.join(base_target_dir_root, job_time_log), "w") as f:
            #     f.write(f"{total_end - total_start:.2f}\n")
            # print(f"\nAll simulations finished in {total_end - total_start:.2f} seconds")

            #         run_foam("pimpleFoam", target_dir)
            #         print(f"Finished simulation for {sim_name} in {target_dir}")

            #         # No need to copy results back or clean generation_dir here

            #     # Clean the generation directory after processing all velocities for the current mesh
            #     print(f"Cleaning generation directory {generation_dir} after processing mesh {mesh}")
            #     run_foam("./Allclean", generation_dir)

    print("All simulations finished.")
