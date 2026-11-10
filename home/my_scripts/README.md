# my_scripts (SAM3D container)

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
