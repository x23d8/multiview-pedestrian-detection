import json
import os

import numpy as np
from torchvision.datasets import VisionDataset


class WaveTrack(VisionDataset):
    """WaveTrack: 5 cameras (Cam1..Cam5), xy-indexing, metres, 2.5 cm cells (MultiviewX convention).

    The BEV grid is the 7.82 x 6.02 m ROI (313 x 241 cells) padded to a multiple of 8 (320 x 248; MVDeTr's deformable world encoder needs it).
    ``positionID`` encodes ``x + y * 313`` (the unpadded width). Prepare the folder with
    data_tools/prepare_wavetrack.py: it fills the missing positionIDs and removes non-annotation files.
    Calibration is read from calibrations/calibration.json (K, R, tvec in metres, undistorted images).
    Cameras use the OpenCV convention (positive depth, z up), unlike MultiviewX, so configs use positive depth.
    """
    POS_WIDTH = 313

    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexing; x,y (w,h) (n_col,n_row)
        self.__name__ = 'WaveTrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [248, 320]  # H,W; N_row,N_col
        # frame ids are 0, 30, ..., 26070 (870 frames). Train/test split is 80/20 for WaveTrack (90/10 for the
        # other datasets): 26071 puts the cut between id 20850 and 20880 -> 696 train frames (even, as
        # EarlyBird's RandomPairSampler needs) and 174 test frames, using every frame.
        self.num_cam, self.num_frame = 5, 26071
        self.train_ratio = 0.8
        # annotated frames are 30 video frames apart (2 fps subset); used for temporal context / tracking ids
        self.frame_step = 30
        # world x,y correspond to w,h
        self.worldcoord_from_worldgrid_mat = np.array([[0.025, 0, 0], [0, 0.025, 0], [0, 0, 1]])
        with open(os.path.join(root, 'calibrations', 'calibration.json'), encoding='utf-8') as stream:
            cameras = {item['name']: item for item in json.load(stream)['cameras']}
        self._cameras = [cameras[f'Cam{index}'] for index in range(1, self.num_cam + 1)]
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            if not camera_folder.startswith('C'):
                continue
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % self.POS_WIDTH
        grid_y = pos // self.POS_WIDTH
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        item = self._cameras[camera_i]
        intrinsic_matrix = np.asarray(item['camera_matrix'], dtype=np.float64)
        extrinsic_matrix = np.hstack([np.asarray(item['R'], dtype=np.float64),
                                      np.asarray(item['tvec_m'], dtype=np.float64).reshape(3, 1)])
        return intrinsic_matrix, extrinsic_matrix
