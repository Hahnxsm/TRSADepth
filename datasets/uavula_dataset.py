import cv2
import numpy as np

from datasets.mono_dataset import MonoDataset
import PIL.Image as pil
import os

from mono_utils import read_array


class UAVulaDataset(MonoDataset):

    def __init__(self, *args, **kwargs):
        super(UAVulaDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.78913, 0, 0.49802, 0],
                           [0, 1.40643, 0.45859, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (512, 288)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return True


    def get_image_path(self, folder, frame_index, side):
        f_str = f"{frame_index:010d}.jpg"
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return os.path.normpath(image_path)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = f"{frame_index:010d}.jpg.geometric.bin"
        depth_path = os.path.join(self.data_path, folder, 'stereo/depth_maps', f_str)
        depth = read_array(depth_path)
        min_depth, max_depth = np.percentile(
            depth, [5, 95]
        )
        # 截断深度时排除空值干扰
        mask = (depth < min_depth) & (depth != 0)
        depth[mask] = min_depth
        # depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth
        if do_flip:
            depth = np.fliplr(depth)
        depth = cv2.resize(depth, (512, 288), interpolation=cv2.INTER_NEAREST)
        return depth
