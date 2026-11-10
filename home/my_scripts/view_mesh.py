#!/usr/bin/env python
"""
Interactive 3D viewer for a mesh file (.obj, .glb, .gltf, .ply, .stl, .off).

    python ~/my_scripts/view_mesh.py                          # newest ~/sam3d_outputs/*/object.glb
    python ~/my_scripts/view_mesh.py ~/sam3d_outputs/mug/object.glb
    python ~/my_scripts/view_mesh.py model.obj --wireframe

Mouse: left-drag = rotate, scroll = zoom, ctrl/shift + drag = pan.
Keys:  w = toggle wireframe, b = toggle back faces, r = reset view, h = full key list (in terminal),
       q / Esc = close.
"""
import argparse
import glob
import os
import sys

import numpy as np

OUTPUTS = os.path.expanduser("~/sam3d_outputs")


def load_mesh(path):
    """Loads any trimesh-readable file into an open3d TriangleMesh (or PointCloud), keeping colours."""
    import open3d as o3d
    import trimesh

    loaded = trimesh.load(path, force="scene")
    geoms = list(loaded.dump()) if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [g for g in geoms if isinstance(g, trimesh.Trimesh)]

    if not meshes:  # e.g. a point-cloud .ply
        pts = [g for g in geoms if isinstance(g, trimesh.PointCloud)]
        if not pts:
            sys.exit(f"No mesh or points found in {path}")
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([p.vertices for p in pts])))
        colors = [p.colors for p in pts if p.colors is not None and len(p.colors) == len(p.vertices)]
        if len(colors) == len(pts):
            pc.colors = o3d.utility.Vector3dVector(np.vstack(colors)[:, :3] / 255.0)
        return pc

    # Textured meshes (e.g. .obj + .mtl + .png) are converted to per-vertex colours.
    for m in meshes:
        if isinstance(m.visual, trimesh.visual.TextureVisuals):
            try:
                m.visual = m.visual.to_color()
            except Exception:  # noqa: BLE001  no usable texture, keep default grey
                pass
    tm = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
                                     o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)))
    colors = getattr(tm.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(tm.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors)[:, :3] / 255.0)
    mesh.compute_vertex_normals()
    return mesh


def show(path, wireframe=False, axes=False):
    import open3d as o3d

    geom = load_mesh(path)
    if isinstance(geom, o3d.geometry.TriangleMesh):
        print(f"{path}: {len(geom.vertices)} vertices, {len(geom.triangles)} triangles")
    else:
        print(f"{path}: point cloud with {len(geom.points)} points")
    geoms = [geom]
    if axes:
        size = 0.3 * np.max(geom.get_max_bound() - geom.get_min_bound())
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size))
    print("Viewer: drag = rotate, scroll = zoom, w = wireframe, q = close.")
    title = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
    o3d.visualization.draw_geometries(geoms, window_name=title, mesh_show_wireframe=wireframe,
                                      mesh_show_back_face=True)


def newest_output():
    files = glob.glob(os.path.join(OUTPUTS, "*", "object.glb"))
    if not files:
        sys.exit(f"No file given and no {OUTPUTS}/*/object.glb found.")
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="mesh file (default: newest SAM 3D output)")
    parser.add_argument("--wireframe", action="store_true", help="draw triangle edges on top")
    parser.add_argument("--axes", action="store_true", help="show an XYZ axis frame (red/green/blue)")
    args = parser.parse_args()

    path = os.path.expanduser(args.path) if args.path else newest_output()
    if not os.path.exists(path):
        sys.exit(f"{path} does not exist.")
    show(path, wireframe=args.wireframe, axes=args.axes)


if __name__ == "__main__":
    main()
