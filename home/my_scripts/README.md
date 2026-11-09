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
