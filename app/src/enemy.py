import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle
import matplotlib.pyplot as plt
from minigrid.core.grid import Grid
from app.utils.point_in_ellipse import point_in_ellipse

class RimWorldEnemy(WorldObj):
    def __init__(self, color='red'):
        super().__init__('ball', color)

    def render(self, img):
        body_poly = point_in_ellipse(0.5, 0.7, 0.35, 0.25)
        fill_coords(img, body_poly, (200, 180, 150)) 

        shirt_poly = point_in_ellipse(0.5, 0.7, 0.35, 0.25)
        fill_coords(img, shirt_poly, (255, 0, 0))

        head_poly = point_in_circle(0.5, 0.35, 0.22)
        fill_coords(img, head_poly, (220, 200, 170))

        left_eye_poly = point_in_circle(0.4, 0.35, 0.05)
        fill_coords(img, left_eye_poly, (20, 20, 20))

        right_eye_poly = point_in_circle(0.6, 0.35, 0.05)
        fill_coords(img, right_eye_poly, (20, 20, 20))



if __name__ == "__main__":
    grid = Grid(3, 3)
    
    enemy = RimWorldEnemy()
    grid.set(1, 1, enemy)
    
    tile_size = 64
    img = grid.render(tile_size=tile_size, agent_pos=None)
    
    plt.imshow(img)
    plt.axis('off') 
    plt.savefig('app/data/media/enemy_render.png', bbox_inches='tight')