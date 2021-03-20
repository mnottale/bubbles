import cv2
import numpy as np

from .effect_renderer import EffectRenderer

def transparentOverlay(src , overlay , pos=(0,0)):
    if pos[0] >= src.shape[1] or pos[1] >= src.shape[0] or pos[0]+overlay.shape[1] <= 0 or pos[1]+overlay.shape[0] <= 0:
        return
    # crop check
    if pos[0] + overlay.shape[1] >= src.shape[1]:
        overlay = overlay[:,:src.shape[1]-pos[0], :]
    if pos[1] + overlay.shape[0] >= src.shape[0]:
        overlay = overlay[:src.shape[0]-pos[1],:,:]
    if pos[0] < 0:
        overlay = overlay[-pos[0]:,:,:]
        pos = (0, pos[1])
    if pos[1] < 0:
        overlay = overlay[:,-pos[1]:,:]
        pos = (pos[0], 0)
    zone=src[pos[1]:pos[1]+overlay.shape[0],pos[0]:pos[0]+overlay.shape[1],:]
    alpha = overlay[:,:,3] / 255.0
    alpha3 = np.zeros(zone.shape, dtype=np.float64)
    alpha3[:,:,0] = alpha
    alpha3[:,:,1] = alpha
    alpha3[:,:,2] = alpha
    alpha3[:,:,3] = alpha
    res = (zone/255.0) * (1.0 - alpha3) + (overlay/255.0)*alpha3
    res = res * 255.0
    src[pos[1]:pos[1]+overlay.shape[0],pos[0]:pos[0]+overlay.shape[1]] = res

def rotate_image(image, angle, scale):
    image_center = (image.shape[0]/2, image.shape[1]/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle * 180.0 / 3.14159265, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[0], image.shape[1]), flags=cv2.INTER_LINEAR)
    return result

class OpenCVEffectRenderer(EffectRenderer):

    def __init__(self):
        super().__init__()
        self._shapes = {
            "circle": self._render_circle,
            "square": self._render_square
        }

    def _render_particle(self, particle, surface, position):
        if particle.shape in self._shapes.keys():
            texture = self._shapes[particle.shape](particle)
        else:
            texture = self._render_texture(particle)
        transparentOverlay(surface, texture, (round(position[0]), round(position[1])))
        return surface

    def _render_texture(self, particle):
        texture = self._textures[particle.shape].copy()
        if particle.colourise:
            overlay = np.zeros(texture.shape, dtype=np.float64)
            overlay[:] = tuple([round(i) for i in particle.colour] + [255])
            overlay = overlay / 255.0
            texture = overlay * texture
        size = round(self.base_size * particle.scale)
        texture = rotate_image(texture, particle.rotation, particle.scale)
        texture = texture * particle.opacity
        return texture

    def _load_texture(self, filename):
        texture = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if texture.shape[2] == 3:
            texture = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2RGBA)
            texture[:,:,3] = 255
        return texture
    def _get_shape_surface(self, particle):
        size = round(self.base_size * particle.scale)
        return np.zeros((size, size, 4), dtype=np.uint8), size

    def _render_circle(self, particle):
        texture, size = self._get_shape_surface(particle)
        cv2.circle(texture, (size/2, size/2), size, list(particle.colour)+[particle.opacity * 255], -1)
        return texture

    def _render_square(self, particle):
        texture, size = self._get_shape_surface(particle)
        texture = texture + list(round(i) for i in list(particle.colour) + [particle.opacity * 255])
        return texture
