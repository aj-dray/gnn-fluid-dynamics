'''
Generates gmsh meshes for flow around an elliptic cylinder.
Config hardcoded in # === INPUTS ===.
Mesh refinement field tuned within createGradedMesh.
Parts of code credited to: https://github.com/cmudrc/FlowDataGeneration/blob/main/mesh.py
'''

import os

from torch_geometric.data.dataset import re

# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases!

from cmath import pi
import numpy as np
import gmsh
import random
import csv
import shutil
import json

def sample_asymmetric_uniform(mean=40, min_val=10, max_val=300):
    """
    Sample a single value from the asymmetric uniform distribution
    """
    # 50% chance to sample from lower half, 50% from upper half
    if np.random.random() < 0.2:
        # Sample from lower half [min_val, mean]
        return np.random.uniform(min_val, mean)
    else:
        # Sample from upper half (mean, max_val]
        return np.random.uniform(mean, max_val)


def write_metadata(directory, filename, obstacle_type, position, radii, h_obstacle, d_obstacle, num_vertices, num_cells):
    meta_file = os.path.join(directory, "meta.csv")
    file_exists = os.path.isfile(meta_file)
    with open(meta_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header if file does not exist
        if not file_exists:
            writer.writerow([
                'mesh_name', 'obstacle_type',
                'position_x', 'position_y', 'position_angle',
                'radii_major', 'radii_minor',
                'h_ref_min', 'h_ref_max',
                'd_obstacle_min', 'd_obstacle_max',
                'num_vertices', 'num_cells'
            ])
        writer.writerow([
            filename,
            obstacle_type,
            float(position[0]), float(position[1]), float(position[2]),
            float(radii[0]), float(radii[1]),
            float(h_obstacle[0]), float(h_obstacle[1]),
            float(d_obstacle[0]), float(d_obstacle[1]),
            int(num_vertices),
            int(num_cells),
        ])

def createGradedMesh(domain_size, obstacle_type, position, radii, h_global, h_obstacle, h_wall, d_obstacle, d_wall, topBot, filename = '', directory = ''):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.Smoothing",      1)    # no global mesh smoothing
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.5)  # no boundary-based interpolation
    gmsh.model.add("mesh")

    p0, p1, p2, p3 = [gmsh.model.occ.addPoint(*pt, 0)
                      for pt in [(0,0,0), (domain_size[0],0,0),
                                 (domain_size[0],domain_size[1],0), (0,domain_size[1],0)]]

    l_bot  = gmsh.model.occ.addLine(p0, p1)             # inlet
    l_out = gmsh.model.occ.addLine(p1, p2)             # lower wall
    l_top = gmsh.model.occ.addLine(p2, p3)             # outlet
    l_in = gmsh.model.occ.addLine(p3, p0)             # upper wall

    outer = gmsh.model.occ.addCurveLoop([l_in, l_bot, l_out, l_top])

    # Create curves
    if obstacle_type == 'circle':
        c = gmsh.model.occ.addCircle(position[0], position[1], 0, radii)
    elif obstacle_type == 'ellipse':
        c = gmsh.model.occ.addEllipse(position[0], position[1], 0,
                                      radii[0], radii[1],
                                      xAxis=[np.cos(position[2]*pi/180),
                                             np.sin(position[2]*pi/180),0],
                                      zAxis=[0,0,1])
    else:
        raise ValueError("body_type must be 'circle' or 'ellipse'")

    # define surface
    inner = gmsh.model.occ.addCurveLoop([c])
    surf  = gmsh.model.occ.addPlaneSurface([outer, inner])
    gmsh.model.occ.synchronize()

    # global coarse mesh size
    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_global)

    # obstacle surface refinement
    h_min_obs, h_max_obs = h_obstacle
    d_min_obs, d_max_obs = d_obstacle
    f_obs_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_obs_dist, "EdgesList", [c])
    f_obs_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_obs_thresh, "InField", f_obs_dist)
    gmsh.model.mesh.field.setNumber(f_obs_thresh, "SizeMin", h_min_obs)
    gmsh.model.mesh.field.setNumber(f_obs_thresh, "SizeMax", h_max_obs)
    gmsh.model.mesh.field.setNumber(f_obs_thresh, "DistMin", d_min_obs)
    gmsh.model.mesh.field.setNumber(f_obs_thresh, "DistMax", d_max_obs)

    # lower‑wall surface refinement  (l_bot)
    h_min_wall, h_max_wall = h_wall
    d_min_wall, d_max_wall = d_wall
    f_wall_bot_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_wall_bot_dist, "CurvesList", [l_bot])
    gmsh.model.mesh.field.setNumber(f_wall_bot_dist, "Sampling", 200)

    f_wall_bot_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_wall_bot_thresh, "InField", f_wall_bot_dist)
    gmsh.model.mesh.field.setNumber(f_wall_bot_thresh, "SizeMin", h_min_wall)
    gmsh.model.mesh.field.setNumber(f_wall_bot_thresh, "SizeMax", h_max_wall)
    gmsh.model.mesh.field.setNumber(f_wall_bot_thresh, "DistMin", d_min_wall)
    gmsh.model.mesh.field.setNumber(f_wall_bot_thresh, "DistMax", d_max_wall)

    # upper wall refinement (l_top)
    f_wall_top_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_wall_top_dist, "CurvesList", [l_top])
    gmsh.model.mesh.field.setNumber(f_wall_top_dist, "Sampling", 200)

    f_wall_top_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_wall_top_thresh, "InField", f_wall_top_dist)
    gmsh.model.mesh.field.setNumber(f_wall_top_thresh, "SizeMin", h_min_wall)
    gmsh.model.mesh.field.setNumber(f_wall_top_thresh, "SizeMax", h_max_wall)
    gmsh.model.mesh.field.setNumber(f_wall_top_thresh, "DistMin", d_min_wall)
    gmsh.model.mesh.field.setNumber(f_wall_top_thresh, "DistMax", d_max_wall)

    # Wake field
    # x0, y0 = position[:2]
    # k_r    = 0.20                  # half‑width sideways
    # k_x    = 0.8                  # length downstream
    # n      = 4                     # tip sharpness
    # h_wake = h_min_obs             # size you want inside the wake
    # h_far  = h_max_obs             # same as far‑field size
    reference = 0.07
    scale_factor = radii[0] / reference
    x0, y0 = position[:2]
    k_r    = 0.25 * scale_factor                 # half‑width sideways
    k_x    = 1.7 * scale_factor                  # length downstream
    n      = 4                     # tip sharpness
    h_wake = h_min_obs * 1.75             # size you want inside the wake
    h_far  = h_max_obs             # same as far‑field size


    f_wake = gmsh.model.mesh.field.add("MathEval")
    expr_wake = (
        f"Step({x0}-x)*{h_far}"                  # upstream → coarse
        f" + (1-Step({x0}-x))*("                 # downstream → tear‑drop
        f"{h_far} + ({h_wake}-{h_far})/"
        f"(1 + ((max(0,x-{x0})/{k_x})^2 + "
        f"((y-{y0})/{k_r})^2)^{n/2}))"
    )
    gmsh.model.mesh.field.setString(f_wake, "F", expr_wake)

    # # combine all fields
    f_combined = gmsh.model.mesh.field.add("Min")
    if topBot == 'noSlip':
        gmsh.model.mesh.field.setNumbers(f_combined, "FieldsList",
                                        [f_wall_bot_thresh, f_wall_top_thresh, f_obs_thresh, f_wake])
    else:
        gmsh.model.mesh.field.setNumbers(f_combined, "FieldsList",
                                        [f_obs_thresh, f_wake])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_combined)


    # label physical groups
    gmsh.model.addPhysicalGroup(1, [l_in],  name='inlet')
    gmsh.model.addPhysicalGroup(1, [l_out], name='outlet')
    gmsh.model.addPhysicalGroup(1, [l_bot, l_top], name='wall')
    gmsh.model.addPhysicalGroup(1, [c], name='obstacle')
    gmsh.model.addPhysicalGroup(2, [surf], name='flow')

    # generate 2D mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize('Netgen', niter=5)
    gmsh.model.occ.synchronize()

    # save 2D mesh
    gmsh.write(os.path.join(directory, filename, 'mesh.msh'))
    gmsh.model.removePhysicalGroups([(2, gmsh.model.getPhysicalGroups(2)[0][1])])

    # save a meta file of stats
    # 1) Nodes
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    num_nodes = len(node_tags)


    # 2D cells (triangles, quads, …)
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    num_cells = sum(len(tags) for tags in elem_tags)

    print(f"Nodes: {num_nodes}")
    print(f"Cells: {num_cells}")

    # write_metadata(os.path.join(directory, "extruded"), filename, obstacle_type, position, radii, h_obstacle, d_obstacle,
    #                num_nodes, num_cells)
    # gmsh.write(os.path.join(directory, "extruded", filename + '.brep'))
    gmsh.model.occ.synchronize()

    dim, tag = 2, surf
    out = gmsh.model.occ.extrude([(dim, tag)], 0, 0, 0.001, numElements=[1], recombine=True)
    gmsh.model.occ.synchronize()

    # relabel physical groups
    vol = next(tag for dim, tag in out if dim == 3)  # Volume
    lateral_surfs = [tag for dim, tag in out[2:] if dim == 2]  # Lateral surfaces
    gmsh.model.addPhysicalGroup(2, [lateral_surfs[0]], name='inlet')  # from l_in
    gmsh.model.addPhysicalGroup(2, [lateral_surfs[2]], name='outlet')  # from l_out
    gmsh.model.addPhysicalGroup(2, [lateral_surfs[3], lateral_surfs[1]], name='walls')  # from l_bot, l_top, c
    gmsh.model.addPhysicalGroup(2, [lateral_surfs[4]], name='obstacle')
    gmsh.model.addPhysicalGroup(2, [tag, 7], name='frontAndBack')  # Bottom and top as 'empty'
    gmsh.model.addPhysicalGroup(3, [vol], name='flow')  # Volume

            # 1) get the volume tag you created
    vol_tag = next(tag for dim, tag in out if dim == 3)

    # 2) query Gmsh for the boundary faces of that volume
    boundary = gmsh.model.getBoundary([(3, vol_tag)], combined=False, oriented=False)

    untagged = []
    for dim, tag in boundary:
        if dim != 2:
            continue
        # fetch the list/array of (dim,tag) physical groups this face belongs to
        pg = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        # explicitly check for emptiness
        if len(pg) == 0:
            untagged.append(tag)

    print(f"Volume {vol_tag} has {len(boundary)} boundary faces.")
    if untagged:
        print(f"  ▶︎ {len(untagged)} faces are untagged: {untagged}")
    else:
        print("  ✔︎ all boundary faces are in a Physical Group.")

    # generate mesh
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(3)

    # save 3D mesh
    gmsh.write(os.path.join(directory, filename, 'mesh_extruded.msh'))
    gmsh.finalize()

    return num_cells, num_nodes


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        raise ValueError("What device are you on?")

    if device == 'mac':
        path = '/Users/adamdray/Code/data/gmsh'
    elif device == 'csd3':
        path = '/home/ajd246/rds/rds-t2-cs181-iLDMbuOsGy8/ajd246/custom/gmsh'


    folder_name = 'ellipseflow6'

    directory = f'{path}/{folder_name}/'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # === INPUTS ===
    ## Independent
    a_min = 0.06
    a_max = 0.15
    Re_min = 50
    # Re_mean = (250-20)/2  # Set your desired mean here
    Re_max = 200 # 250
    nu = 0.001
    aspect_ratio = 1.25
    refinement = 1 / 15
    angle_max = 90
    angle_min = -90
    number_of_meshes = 1000 #1000 #300
    rng_seed = 1
    topBotType = 'noSlip'
    # ===

    ## Dependent
    D_min = 2 * a_min
    D_max = 2 * a_max
    v_max = Re_max * nu / D_min
    domain_size = np.array([a_max * 20, a_max * 10])

    h_min = D_max * refinement #ERR: should be D_min here
    h_max = D_min / 2
    dt = h_min / (2 * v_max)
    print(dt)

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    distribution = random.uniform
    # skew_distribution = sample_asymmetric_uniform(Re_mean, Re_min, Re_max)
    Re_list = []

    for j in range(number_of_meshes):

        mesh_name = f"mesh_{j}"
        os.makedirs(f"{directory}/{mesh_name}", exist_ok=True)


        x_coordinate = distribution(domain_size[1] / 2, domain_size[1] / 2)
        y_coordinate = distribution(domain_size[1] / 2, domain_size[1] / 2)
        a = distribution(a_min, a_max)
        b = a / aspect_ratio
        D = 2*a
        long_axis = np.max([a, b])
        short_axis = np.min([a, b])
        angle = distribution(angle_min, angle_max)
        d_obstacle = np.array([D / 10,  D * 2])
        h_obstacle = [h_min, h_max]
        h_wall = [h_min, h_max]
        d_wall = np.array([0.01, 0.15])
        # Re = sample_asymmetric_uniform(Re_mean, Re_min, Re_max)
        Re = distribution(Re_min, Re_max)
        Re_list.append(Re)
        u = Re * nu / (D)

        num_cells, num_nodes = createGradedMesh(domain_size, 'ellipse', np.array([x_coordinate, y_coordinate, angle]), np.array([a, b]), h_max, h_obstacle, h_wall, d_obstacle, d_wall, topBotType, filename=mesh_name, directory=directory)

        meta = {
            'geometry' :{
                'position': np.array([x_coordinate, y_coordinate]),
                'radius': np.array([long_axis, short_axis]),
                'aspect_ratio': aspect_ratio,
                'angle': angle,
                'num_vertices': num_nodes,
                'num_cells': num_cells,
            },
            "boundary_conditions" :{
                'inlet': {'field': 'velocity', 'value': u},
                'outlet': {'field': 'pressure', 'value': 0.0},
                'walls': {'type': topBotType},
                'obstacle': {'type': 'noSlip'},
                'frontAndBack': {'type': 'empty'}
            },
            'physics' :{
                'nu': nu,
                'Re': Re,
                'dt': dt
            },
        }

        with open(os.path.join(directory, f"mesh_{j}/meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print("Re: ", Re_list)
