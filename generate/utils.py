import subprocess
import os
import sys
import shutil


def run_foam(commands, dir, on_mac=True):
    """Run OpenFOAM commands in the specified directory."""
    if on_mac:
        subprocess.run(["/opt/homebrew/bin/openfoam", "-c", commands], cwd=dir, check=True)
    else:
        subprocess.run(commands, cwd=dir, check=True, shell=True)


def run_foam_parallel(commands, dir, nprocs=4, on_mac=True):
    """Run OpenFOAM commands in parallel using mpirun."""
    if on_mac:
        parallel_cmd = f"mpirun -np {nprocs} {commands}"
        subprocess.run(["/opt/homebrew/bin/openfoam", "-c", parallel_cmd], cwd=dir, check=True)
    else:
        parallel_cmd = f"mpirun -np {nprocs} {commands}"
        subprocess.run(parallel_cmd, cwd=dir, check=True, shell=True)


def update_decompose_par_dict(case_dir, nprocs, on_mac=True):
    """Update decomposeParDict with the specified number of processors and optimal layout."""
    import math
    
    # Calculate optimal processor layout (nx, ny, nz) for the given number of processors
    # For 2D case (nz=1), we need nx * ny = nprocs
    nz = 1
    
    # Find factors of nprocs that are closest to square
    best_nx, best_ny = 1, nprocs
    min_ratio = float('inf')
    
    for nx in range(1, int(math.sqrt(nprocs)) + 1):
        if nprocs % nx == 0:
            ny = nprocs // nx
            ratio = max(nx, ny) / min(nx, ny)
            if ratio < min_ratio:
                min_ratio = ratio
                best_nx, best_ny = nx, ny
    
    # Update decomposeParDict
    decompose_dict = os.path.join(case_dir, "system", "decomposeParDict")
    entries = {
        "numberOfSubdomains": str(nprocs),
        "simpleCoeffs.n": f"({best_nx} {best_ny} {nz})"
    }
    modify_foamDict(decompose_dict, entries, on_mac=on_mac)
    print(f"Updated decomposeParDict: {nprocs} cores with layout ({best_nx} {best_ny} {nz})")


def modify_time_step(case_dir, dt, on_mac=True):
    """Modify only the deltaT (time step) in controlDict."""
    control_dict = os.path.join(case_dir, "system", "controlDict")
    entries = {"deltaT": f"{dt}"}
    modify_foamDict(control_dict, entries, on_mac=on_mac)
    print(f"Updated deltaT to {dt} in {control_dict}")


def modify_foamDict(file_path, entries, on_mac=True):
    """Modify entries in an OpenFOAM dictionary file using foamDictionary."""
    # Start building the command list
    if not entries:
            return

    # Build a combined command with all modifications
    commands = []
    for entry, value in entries.items():
        # Add single quotes around the value to handle spaces/special characters
        quoted_value = f"'{value}'"
        commands.append(f"foamDictionary {file_path} -entry {entry} -set {quoted_value}")

    # Join all commands with && to execute them in a single process
    combined_cmd = " && ".join(commands)

    # Execute all commands at once
    run_foam(combined_cmd, os.path.dirname(file_path), on_mac=on_mac)


def update_boundary_file(file_path):
    """
    Modify the OpenFOAM boundary file to change patch types for specific sections.
    Changes 'frontAndBack' to 'empty', 'walls' and 'obstacle' to 'wall'.
    """
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flag to track when we're inside the boundary patch section
    in_boundary_section = False

    # Process each line
    in_empty_section = False
    in_wall_section = False
    in_obstacle_section = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Detect when we enter the boundary patch section
        if (line_stripped == "frontAndBack") and lines[i+1].strip() == "{":
            in_empty_section = True
            continue

        # When in boundary section, replace patch with empty
        if in_empty_section:
            if "type" in line_stripped and "patch" in line_stripped:
                lines[i] = line.replace("patch", "empty")
            elif "physicalType" in line_stripped and "patch" in line_stripped:
                lines[i] = line.replace("patch", "empty")
                                    # Exit the boundary section when we reach its closing bracket
            if line_stripped == "}":
                in_empty_section = False

                # Detect when we enter the boundary patch section
        if line_stripped == "walls" and lines[i+1].strip() == "{":
            in_wall_section = True
            continue

        if in_wall_section:
                    # Update wall to type wall
            if "type" in line_stripped and "patch" in line_stripped: # else use "walls"
                lines[i] = line.replace("patch", "wall")
            elif "physicalType" in line_stripped and "patch" in line_stripped:
                lines[i] = line.replace("patch", "wall")

                            # Exit the boundary section when we reach its closing bracket
            if line_stripped == "}":
                in_wall_section = False

        if line_stripped == "obstacle" and lines[i+1].strip() == "{":
            in_obstacle_section = True
            continue

        if in_obstacle_section:
                    # Update wall to type wall
            if "type" in line_stripped and "patch" in line_stripped: # else use "walls"
                lines[i] = line.replace("patch", "wall")
            elif "physicalType" in line_stripped and "patch" in line_stripped:
                lines[i] = line.replace("patch", "wall")

                            # Exit the boundary section when we reach its closing bracket
            if line_stripped == "}":
                in_obstacle_section = False
        
        

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def gmsh_to_foam(filepath, case_dir, on_mac):
    """Convert a GMSH mesh to OpenFOAM format and update boundary and mesh files."""
    # Set dataup set directory
    name_mesh = os.path.basename(filepath).split(".")[0]
    dir_case = case_dir

    # Remove any previous .msh files in the case directory
    if os.path.isdir(dir_case):
        for f in os.listdir(dir_case):
            if f.endswith(".msh"):
                try:
                    os.remove(os.path.join(dir_case, f))
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")


    # Copy mesh file to case directory
    fpath = f"{filepath}/mesh.msh"
    shutil.copy2(fpath, dir_case)
    fpath = f"{filepath}/mesh_extruded.msh"
    shutil.copy2(fpath, dir_case)

    # Convert the mesh using gmshToFoam
    print("Running gmshToFoam...")
    run_foam(f"gmshToFoam mesh_extruded.msh", dir_case, on_mac=on_mac)

    # Modify boundary file
    boundary_file = os.path.join(dir_case, "constant", "polyMesh", "boundary")
    update_boundary_file(boundary_file)

    # Run checkMesh
    print("Running checkMesh...")
    run_foam("checkMesh -allTopology -allGeometry | tee checkMesh.log", dir_case, on_mac=on_mac)

    # make foam.foam for vis
    subprocess.run(["touch", "foam.foam"], cwd=dir_case)

    print("Mesh conversion and check completed successfully for", name_mesh)
