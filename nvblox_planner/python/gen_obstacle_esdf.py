import argparse
import os
import sys
import numpy as np
import open3d as o3d
from math import sqrt

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_ply', type=str, help='input.ply')
  parser.add_argument('--output_npy', type=str, help='output.npy')
  parser.add_argument('--obstacle_distance', type=float, help='0.1')
  args = parser.parse_args()
  print("Arguments:\n{}".format('\n'.join(
      ['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

  if args.input_ply:
    # ptcloud_esdf = o3d.t.io.read_point_cloud(args.input_ply)
    # print(ptcloud_esdf)
    # xyz = np.asarray(ptcloud_esdf.point["points"])
    # intensity = np.asarray(ptcloud_esdf.point["intensity"])
    # print(xyz.shape)
    # print(intensity.shape)

    # xyz = xyz[abs(intensity) <= args.obstacle_distance]
    # intensity = intensity[abs(intensity) <= args.obstacle_distance]
    # print(xyz.shape)
    # print(intensity.shape)

    pcd = o3d.io.read_point_cloud(args.input_ply)
    print(pcd)
    print(pcd.points)
    print(pcd.normals)
    print(pcd.colors)
