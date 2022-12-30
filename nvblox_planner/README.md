# NVBlox Planar

1. <code>load_mesh_map</code>
  * This file process the 3D mesh map and generate a two-layer 2D cost map
  * Map color with the corresponding label: [semantic_kitti_all.yaml](docs/semantic_kitti_all.yaml)
  * Usage:
    * Download test data: 
      ```
      http://gofile.me/72EEc/dVWI3QZil
      http://gofile.me/72EEc/fJb6LvLhW
      ```
    * Run ```python3 load_mesh_map.py --ground_mesh_map ground_mesh.ply --nonground_mesh_map nonground_mesh.ply ```