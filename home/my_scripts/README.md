# my_scripts (SAM3D container)

## scan_object.py: hands-free RealSense scan -> metric SAM 3D mesh (no clicking)
```
python ~/my_scripts/scan_object.py mug              # 3 s countdown window, then fully automatic
python ~/my_scripts/scan_object.py mug --input ~/my_scripts/objects/mug   # redo from the saved capture
```
One object on a plain surface, fully in view. The object box comes from the depth: big planes
(table, walls, box sides) are removed and the biggest central blob not touching the image edge wins
(falls back to "differs from the background colour"). SAM 2 makes the mask, SAM 3D the mesh, using the
RealSense depth as its point map. The mesh is scaled to metres (from the depth when the object's longest
side is seen side-on, else SAM 3D's own scale; `--scale depth|sam3d` overrides) and simplified to ~20k triangles.
Saved in `~/my_scripts/objects/<name>/`: `mesh.obj` / `mesh.glb` (metres, for FoundationPose),
`preview.png` (mask + auto box), `info.json` (size, scale source), the capture, raw SAM 3D outputs.
Options: `--countdown 0`, `--no-window`, `--box-method depth|color`, `--view`, `--fps 15`.

## track_object.py: live 6-DoF pose with FoundationPose (conda env `foundationpose`)
```
conda activate foundationpose
python ~/my_scripts/track_object.py mug              # mesh from ~/my_scripts/objects/mug/
```
Finds the object automatically on the first frame (same as scan_object), registers its pose (~2 s), then
tracks at ~60 Hz, drawing the 3D bounding box and XYZ axes (red x, green y, blue z).
Keys: **r** re-detect, **a** axes, **b** box, **q** quit. Re-registers by itself if tracking is lost.
Options: `--record out.mp4`, `--save-poses poses.txt`, `--no-window --frames 300`.
Symmetric objects (cans, bottles) have no defined spin about their axis, so that angle can wander.

## capture_to_3d.py: RealSense photo -> SAM 2 mask -> SAM 3D model
```
python ~/my_scripts/capture_to_3d.py                 # add --name mug to name the output folder
```
1. Live camera window: point at the object, press **SPACE** (q quits).
2. Drag a box around the object, press **ENTER**.
3. Check the green mask: **ENTER** accepts, any other key lets you redraw the box.
4. Wait ~1 min (SAM 3D loads ~20 s, builds ~20-40 s), then a 3D viewer opens (drag = rotate, scroll = zoom, q = close).

Saved in `~/sam3d_outputs/<name>/`: `object.glb` (coloured mesh), `object.ply` (gaussian splat),
`photo.png`, `mask.png`, `mask_preview.png`.

Options: `--fps 15` (if 30 fails on USB 2), `--image photo.png` (use a file instead of the camera),
`--box X1 Y1 X2 Y2` (skip drawing the box), `--no-preview`, `--no-view`.

Tips: one object, fully in view, not cut off by the image edge, decent lighting. SAM 3D guesses the
hidden back side, so the side facing the camera comes out best. The model's size/scale is only roughly right.

## live_view.py: live RealSense view
```
python ~/my_scripts/live_view.py            # add --depth for colour + depth, --fps 15 on USB 2
```
Runs until **Ctrl+C** in the terminal (q / Esc in the window also work).

## view_mesh.py: interactive mesh viewer
```
python ~/my_scripts/view_mesh.py                               # newest ~/sam3d_outputs/*/object.glb
python ~/my_scripts/view_mesh.py path/to/model.obj --wireframe  # any .obj/.glb/.ply/.stl
```
Drag = rotate, scroll = zoom, ctrl/shift + drag = pan, w = wireframe, r = reset view, q = close.
`--axes` adds an XYZ frame. Textured .obj files are shown with per-vertex colours.

## image_to_primitive.py: RealSense RGB-D photo -> cylinder or box primitive
```
python ~/my_scripts/image_to_primitive.py --name mug          # live view, SPACE takes the photo
python ~/my_scripts/image_to_primitive.py --input ~/sam3d_outputs/mug   # re-run on a saved capture
```
One upright object on a table, plain background. A camera looking down at ~20-60 deg works best (top + sides visible).
1. Finds the table plane in the depth, the object is the biggest blob above it (SAM 2 sharpens the mask).
2. Cylinder or box? Box side normals point in 1-2 directions 90 deg apart, a cylinder's are spread out;
   a box's top fills its bounding rectangle, a disk only ~79% of it. Printed as `squareness` (0 = cylinder, 1 = box).
3. Fits height (from the table), radius or box size + orientation. Hidden sizes are assumed
   (box seen face-on: square footprint, override with `--box-depth 0.05`).

Saved in `~/sam3d_outputs/<name>/`: `primitive.json` (type, size in m, 4x4 camera->object pose),
`primitive.obj` (centred, z up), `primitive_in_camera.obj`, `primitive_overlay.png`, `mask.png`,
`object_points.ply`, plus the capture (`color.png`, `depth.png`, `camera.json`).
Options: `--force cylinder|box`, `--no-sam`, `--no-view`, `--fps 15`, `--max-depth 2.0`.
