"""
Simple mesh refinement on an existing gmsh mesh.
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from typing import Optional

def count_triangles_in_mesh(mesh_file: str) -> int:
    """Count the number of triangles in a mesh file."""
    try:
        import meshio
        mesh = meshio.read(mesh_file)

        triangle_count = 0
        for cell in mesh.cells:
            if hasattr(cell, 'type') and hasattr(cell, 'data'):
                # New meshio API
                if cell.type == "triangle":
                    triangle_count = len(cell.data)
                    break
            else:
                # Old meshio API
                cell_type, cell_data = cell
                if cell_type == "triangle":
                    triangle_count = len(cell_data)
                    break

        return triangle_count
    except ImportError:
        print("meshio not available, using gmsh to count triangles")
        return count_triangles_gmsh(mesh_file)

def count_triangles_gmsh(mesh_file: str) -> int:
    """Count triangles using gmsh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

    try:
        gmsh.open(mesh_file)
        # Get all 2D elements (triangles)
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)
        triangle_count = 0
        for i, elem_type in enumerate(element_types):
            if elem_type == 2:  # Triangle element type in gmsh
                triangle_count += len(element_tags[i])
        return triangle_count
    finally:
        gmsh.finalize()


def refine_mesh_gmsh(input_file: str, refinement_levels: int = 1) -> str:
    """Refine mesh using gmsh and return output filename."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.open(input_file)

        # Perform refinement
        for _ in range(refinement_levels):
            gmsh.model.mesh.refine()

        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_{refinement_levels}.msh"
        print(output_file)

        gmsh.write(output_file)
        return output_file

    finally:
        gmsh.finalize()


def get_mesh_triangles(mesh_file: str):
    """Extract triangle coordinates from mesh file."""
    try:
        import meshio
        mesh = meshio.read(mesh_file)

        triangles = None
        for cell in mesh.cells:
            if hasattr(cell, 'type') and hasattr(cell, 'data'):
                if cell.type == "triangle":
                    triangles = cell.data
                    break
            else:
                cell_type, cell_data = cell
                if cell_type == "triangle":
                    triangles = cell_data
                    break

        if triangles is None:
            return None, None

        # Get 2D coordinates (ignore z if present)
        points_2d = mesh.points[:, :2]
        return points_2d, triangles

    except ImportError:
        print("meshio not available for visualization")
        return None, None


def visualize_meshes(original_file: str, refined_file: str):
    """Create overlay visualization of original and refined meshes."""

    # Get triangle data
    orig_points, orig_triangles = get_mesh_triangles(original_file)
    ref_points, ref_triangles = get_mesh_triangles(refined_file)

    if orig_points is None or ref_points is None:
        print("Cannot visualize - meshio not available or no triangles found")
        return

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original mesh
    ax1.set_title('Original Mesh')
    ax1.triplot(orig_points[:, 0], orig_points[:, 1], orig_triangles, 'b-', linewidth=0.8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot refined mesh
    ax2.set_title('Refined Mesh')
    ax2.triplot(ref_points[:, 0], ref_points[:, 1], ref_triangles, 'r-', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot overlay
    ax3.set_title('Overlay Comparison')
    ax3.triplot(orig_points[:, 0], orig_points[:, 1], orig_triangles, 'b-',
                linewidth=1.2, alpha=0.7, label='Original')
    ax3.triplot(ref_points[:, 0], ref_points[:, 1], ref_triangles, 'r-',
                linewidth=0.3, alpha=0.8, label='Refined')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def refine_and_compare(input_file: str, levels: int = 1):
    """Main function to refine mesh and show comparison."""

    print(f"=== Mesh Refinement Comparison ===")
    print(f"Input file: {input_file}")
    print(f"Refinement levels: {levels}")

    # Count triangles in original mesh
    orig_triangles = count_triangles_in_mesh(input_file)
    print(f"Original mesh triangles: {orig_triangles}")

    # Refine the mesh
    print(f"Refining mesh...")
    refined_file = refine_mesh_gmsh(input_file, levels)

    # Count triangles in refined mesh
    refined_triangles = count_triangles_in_mesh(refined_file)
    print(f"Refined mesh triangles: {refined_triangles}")

    # Calculate refinement factor
    if orig_triangles > 0:
        factor = refined_triangles / orig_triangles
        print(f"Refinement factor: {factor:.1f}x")

    print(f"Refined mesh saved as: {refined_file}")
    print()

    # Create visualization
    print("Creating visualization...")
    visualize_meshes(input_file, refined_file)

    return refined_file

if __name__ == "__main__":

    levels=1
    input_mesh = "/Users/adamdray/Code/data/gmsh/laminar_ellipses_mini2/mesh_0/mesh_1.msh"

    refine_and_compare(input_mesh, levels)
