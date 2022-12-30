'''
Usage: python3 load_mesh_map.py \
  --ground_mesh_map ground_mesh.ply \
  --nonground_mesh_map nonground_mesh.ply 
'''

import argparse
import open3d as o3d
import numpy as np
from semantickitti_label import get_semantickitti_label_from_color

def parse_mesh_map(mesh_map):
  pts = np.asarray(mesh_map.vertices) # [N, 3], float
  color = np.asarray(np.asarray(mesh_map.vertex_colors) * 255).astype(int) # [N, 3], uint8
  label = np.zeros((pts.shape[0], 1), dtype=np.int32)
  for i in range(pts.shape[0]):
    label[i] = get_semantickitti_label_from_color(color[i][:])
  return pts, color, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_mesh_map', type=str, help='path to ground_mesh_map')
    parser.add_argument('--nonground_mesh_map', type=str, help='path to nonground_mesh_map')
    args = parser.parse_args()
    print("Arguments:\n{}".format('\n'.join(
        ['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

    ground_mesh_map = o3d.io.read_triangle_mesh(args.ground_mesh_map)
    print(ground_mesh_map)

    nonground_mesh_map = o3d.io.read_triangle_mesh(args.nonground_mesh_map)
    print(nonground_mesh_map)

    ### parse mesh map data
    ground_pts, ground_color, ground_label = parse_mesh_map(ground_mesh_map)
    nonground_pts, nonground_color, nonground_label = parse_mesh_map(nonground_mesh_map)
