import numpy as np

def get_semantickitti_label_from_color(color):
  """
   Args:
      color:

   Returns:
      semantickitti_label:
  """
  semantickitti_label = 0

  if (color[0] == 255 and color[1] == 0 and color[2] == 255):
    semantickitti_label = 40 # road
  elif (color[0] == 255 and color[1] == 150 and color[2] == 255):
    semantickitti_label = 44 # parking
  elif (color[0] == 75 and color[1] == 0 and color[2] == 75):
    semantickitti_label = 48 # sidewalk
  elif (color[0] == 80 and color[1] == 240 and color[2] == 150):
    semantickitti_label = 72 # terrain

  return semantickitti_label