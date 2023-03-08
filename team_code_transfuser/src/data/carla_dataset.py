import os
import ujson
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io

from src.config import GlobalConfig
from .utils import align, crop_image_cv2, crop_seg, decode_pil_to_npy, draw_target_point, get_depth, get_waypoints, lidar_bev_cam_correspondences, lidar_to_histogram_features, load_crop_bev_npy, parse_labels, scale_image_cv2, scale_seg, transform_waypoints


class CARLA_Data(Dataset):
    """
    Iteration yields dict with keys:
        rgb: RGB image from cameras
        bev:  BEV image from lidar
        depth: Depth image from simulator
        semantic: Semantic segmentation image from simulator
    """

    def __init__(self, root, config: GlobalConfig, shared_dict=None):

        self.seq_len = np.array(config.seq_len)
        assert (config.img_seq_len == 1)
        self.pred_len = np.array(config.pred_len)

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)
        self.multitask = np.array(config.multitask)
        self.data_cache = shared_dict
        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.use_point_pillars = np.array(config.use_point_pillars)
        self.max_lidar_points = np.array(config.max_lidar_points)
        self.backbone = np.array(config.backbone).astype(np.string_)
        self.inv_augment_prob = np.array(config.inv_augment_prob)
        
        self.converter = np.uint8(config.converter)

        self.rgbs = []
        self.topdowns = []
        self.depths = []
        self.semantics = []
        self.lidars = []
        self.labels = []
        self.measurements = []

        for sub_root in tqdm(root, file=sys.stdout):
            sub_root = Path(sub_root)

            # list sub-directories in root
            root_files = os.listdir(sub_root)
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
            for route in routes:
                route_dir = sub_root / route
                num_seq = len(os.listdir(route_dir / "lidar"))

                # ignore the first two and last two frame
                for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                    # load input seq and pred seq jointly
                    rgb = []
                    topdown = []
                    depth = []
                    semantic = []
                    lidar = []
                    label = []
                    measurement= []
                    # Loads the current (and past) frames (if seq_len > 1)
                    for idx in range(self.seq_len):
                        rgb.append(route_dir / "rgb" / ("%04d.png" % (seq + idx)))
                        topdown.append(route_dir / "topdown" / ("encoded_%04d.png" % (seq + idx)))
                        depth.append(route_dir / "depth" / ("%04d.png" % (seq + idx)))
                        semantic.append(route_dir / "semantics" / ("%04d.png" % (seq + idx)))
                        lidar.append(route_dir / "lidar" / ("%04d.npy" % (seq + idx)))
                        measurement.append(route_dir / "measurements" / ("%04d.json"%(seq+idx)))

                    # Additionally load future labels of the waypoints
                    for idx in range(self.seq_len + self.pred_len):
                        label.append(route_dir / "label_raw" / ("%04d.json" % (seq + idx)))

                    self.rgbs.append(rgb)
                    self.topdowns.append(topdown)
                    self.depths.append(depth)
                    self.semantics.append(semantic)
                    self.lidars.append(lidar)
                    self.labels.append(label)
                    self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.rgbs         = np.array(self.rgbs        ).astype(np.string_)
        self.topdowns     = np.array(self.topdowns    ).astype(np.string_)
        self.depths       = np.array(self.depths      ).astype(np.string_)
        self.semantics    = np.array(self.semantics   ).astype(np.string_)
        self.lidars       = np.array(self.lidars      ).astype(np.string_)
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        print("Loading %d lidars from %d folders"%(len(self.lidars), len(root)))

    def __len__(self):
        """Returns the length of the dataset. """
        return self.lidars.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        cv2.setNumThreads(0) # Disable threading because the data loader will already split in threads.

        data = dict()
        backbone = str(self.backbone, encoding='utf-8')

        rgbs = self.rgbs[index]
        lidars = self.lidars[index]

        topdowns = self.topdowns[index]
        depths = self.depths[index]
        semantics = self.semantics[index]
        labels = self.labels[index]
        measurements = self.measurements[index]

        # load measurements
        loaded_images = []
        loaded_bevs = []
        loaded_depths = []
        loaded_semantics = []
        loaded_lidars = []
        loaded_labels = []
        loaded_measurements = []

        if(backbone == 'geometric_fusion'):
            loaded_lidars_raw = []

        # Because the strings are stored as numpy byte objects we need to convert them back to utf-8 strings
        # Since we also load labels for future timesteps, we load and store them separately
        for i in range(self.seq_len+self.pred_len):
            if ((not (self.data_cache is None)) and (str(labels[i], encoding='utf-8') in self.data_cache)):
                    labels_i = self.data_cache[str(labels[i], encoding='utf-8')]
            else:

                with open(str(labels[i], encoding='utf-8'), 'r') as f2:
                    labels_i = ujson.load(f2)

                if not self.data_cache is None:
                    self.data_cache[str(labels[i], encoding='utf-8')] = labels_i

            loaded_labels.append(labels_i)


        for i in range(self.seq_len):
            if not self.data_cache is None and str(measurements[i], encoding='utf-8') in self.data_cache:
                    measurements_i, images_i, lidars_i, lidars_raw_i, bevs_i, depths_i, semantics_i = self.data_cache[str(measurements[i], encoding='utf-8')]
                    images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
                    depths_i = cv2.imdecode(depths_i, cv2.IMREAD_UNCHANGED)
                    semantics_i = cv2.imdecode(semantics_i, cv2.IMREAD_UNCHANGED)
                    bevs_i.seek(0) # Set the point to the start of the file like object
                    bevs_i = np.load(bevs_i)['arr_0']
            else:
                with open(str(measurements[i], encoding='utf-8'), 'r') as f1:
                    measurements_i = ujson.load(f1)

                lidars_i = np.load(str(lidars[i], encoding='utf-8'), allow_pickle=True)[1]  # [...,:3] # lidar: XYZI
                if (backbone == 'geometric_fusion'):
                    lidars_raw_i = np.load(str(lidars[i], encoding='utf-8'), allow_pickle=True)[1][..., :3]  # lidar: XYZI
                else:
                    lidars_raw_i = None
                lidars_i[:, 1] *= -1

                images_i = cv2.imread(str(rgbs[i], encoding='utf-8'), cv2.IMREAD_COLOR)
                if(images_i is None):
                    print("Error loading file: ", str(rgbs[i], encoding='utf-8'))
                images_i = scale_image_cv2(cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB), self.scale)

                bev_array = cv2.imread(str(topdowns[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                bev_array = cv2.cvtColor(bev_array, cv2.COLOR_BGR2RGB)
                if (bev_array is None):
                    print("Error loading file: ", str(topdowns[i], encoding='utf-8'))
                bev_array = np.moveaxis(bev_array, -1, 0)
                bevs_i = decode_pil_to_npy(bev_array).astype(np.uint8)
                if self.multitask:
                    depths_i = cv2.imread(str(depths[i], encoding='utf-8'), cv2.IMREAD_COLOR)
                    if (depths_i is None):
                        print("Error loading file: ", str(depths[i], encoding='utf-8'))
                    depths_i = scale_image_cv2(cv2.cvtColor(depths_i, cv2.COLOR_BGR2RGB), self.scale)

                    semantics_i = cv2.imread(str(semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                    if (semantics_i is None):
                        print("Error loading file: ", str(semantics[i], encoding='utf-8'))
                    semantics_i = scale_seg(semantics_i, self.scale)
                else:
                    depths_i = None
                    semantics_i = None

                if not self.data_cache is None:
                    # We want to cache the images in png format instead of uncompressed, to reduce memory usage
                    result, compressed_imgage = cv2.imencode('.png', images_i)
                    result, compressed_depths = cv2.imencode('.png', depths_i)
                    result, compressed_semantics = cv2.imencode('.png', semantics_i)
                    compressed_bevs = io.BytesIO()  # bev has 2 channels which does not work with png compression so we use generic numpy in memory compression
                    np.savez_compressed(compressed_bevs, bevs_i)
                    self.data_cache[str(measurements[i], encoding='utf-8')] = (measurements_i, compressed_imgage, lidars_i, lidars_raw_i, compressed_bevs, compressed_depths, compressed_semantics)

            loaded_images.append(images_i)
            loaded_bevs.append(bevs_i)
            loaded_depths.append(depths_i)
            loaded_semantics.append(semantics_i)
            loaded_lidars.append(lidars_i)
            loaded_measurements.append(measurements_i)
            if (backbone == 'geometric_fusion'):
                loaded_lidars_raw.append(lidars_raw_i)

        labels = loaded_labels
        measurements = loaded_measurements

        # load image, only use current frame
        # augment here
        crop_shift = 0
        degree = 0
        rad = np.deg2rad(degree)
        do_augment = self.augment and random.random() > self.inv_augment_prob
        if do_augment:
            degree = (random.random() * 2. - 1.) * self.aug_max_rotation
            rad = np.deg2rad(degree)
            crop_shift = degree / 60 * self.img_width / self.scale # we scale first

        images_i = loaded_images[self.seq_len-1]
        images_i = crop_image_cv2(images_i, crop=self.img_resolution, crop_shift=crop_shift)

        bevs_i = load_crop_bev_npy(loaded_bevs[self.seq_len-1], degree)
        
        data['rgb'] = images_i
        data['bev'] = bevs_i

        if self.multitask:
            depths_i = loaded_depths[self.seq_len-1]
            depths_i = get_depth(crop_image_cv2(depths_i, crop=self.img_resolution, crop_shift=crop_shift))

            semantics_i = loaded_semantics[self.seq_len-1]
            semantics_i = self.converter[crop_seg(semantics_i, crop=self.img_resolution, crop_shift=crop_shift)]

            data['depth'] = depths_i
            data['semantic'] = semantics_i

        # need to concatenate seq data here and align to the same coordinate
        lidars = []
        if (backbone == 'geometric_fusion'):
            lidars_raw = []
        if (self.use_point_pillars == True):
            lidars_pillar = []

        for i in range(self.seq_len):
            lidar = loaded_lidars[i]
            # transform lidar to lidar seq-1
            lidar = align(lidar, measurements[i], measurements[self.seq_len-1], degree=degree)
            lidar_bev = lidar_to_histogram_features(lidar)
            lidars.append(lidar_bev)

            if (backbone == 'geometric_fusion'):
                # We don't align the raw LiDARs for now
                lidar_raw = loaded_lidars_raw[i]
                lidars_raw.append(lidar_raw)

            if (self.use_point_pillars == True):
                # We want to align the LiDAR for the point pillars, but not voxelize them
                lidar_pillar = deepcopy(loaded_lidars[i])
                lidar_pillar = align(lidar_pillar, measurements[i], measurements[self.seq_len-1], degree=degree)
                lidars_pillar.append(lidar_pillar)

        # NOTE: This flips the ordering of the LiDARs since we only use 1 it does nothing. Can potentially be removed.
        lidar_bev = np.concatenate(lidars[::-1], axis=0)
        if (backbone == 'geometric_fusion'):
            lidars_raw = np.concatenate(lidars_raw[::-1], axis=0)
        if (self.use_point_pillars == True):
            lidars_pillar = np.concatenate(lidars_pillar[::-1], axis=0)

        if (backbone == 'geometric_fusion'):
            curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(lidars_raw), debug=False)


        # ego car is always the first one in label file
        ego_id = labels[self.seq_len-1][0]['id']

        # only use label of frame 1
        bboxes = parse_labels(labels[self.seq_len-1], rad=-rad)
        waypoints = get_waypoints(labels[self.seq_len-1:], self.pred_len+1)
        waypoints = transform_waypoints(waypoints)

        # save waypoints in meters
        filtered_waypoints = []
        for id in list(bboxes.keys()) + [ego_id]:
            waypoint = []
            for matrix, flag in waypoints[id][1:]:
                waypoint.append(matrix[:2, 3])
            filtered_waypoints.append(waypoint)
        waypoints = np.array(filtered_waypoints)

        label = []
        for id in bboxes.keys():
            label.append(bboxes[id])
        label = np.array(label)
        
        # padding
        label_pad = np.zeros((20, 7), dtype=np.float32)
        ego_waypoint = waypoints[-1]

        # for the augmentation we only need to transform the waypoints for ego car
        degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                              [-np.sin(rad), np.cos(rad)]])
        ego_waypoint = (degree_matrix @ ego_waypoint.T).T

        if label.shape[0] > 0:
            label_pad[:label.shape[0], :] = label

        if(self.use_point_pillars == True):
            # We need to have a fixed number of LiDAR points for the batching to work, so we pad them and save to total amound of real LiDAR points.
            fixed_lidar_raw = np.empty((self.max_lidar_points, 4), dtype=np.float32)
            num_points = min(self.max_lidar_points, lidars_pillar.shape[0])
            fixed_lidar_raw[:num_points, :4] = lidars_pillar
            data['lidar_raw'] = fixed_lidar_raw
            data['num_points'] = num_points

        if (backbone == 'geometric_fusion'):
            data['bev_points'] = curr_bev_points
            data['cam_points'] = curr_cam_points

        data['lidar'] = lidar_bev
        data['label'] = label_pad
        data['ego_waypoint'] = ego_waypoint

        # other measurement
        # do you use the last frame that already happend or use the next frame?
        data['steer'] = measurements[self.seq_len-1]['steer']
        data['throttle'] = measurements[self.seq_len-1]['throttle']
        data['brake'] = measurements[self.seq_len-1]['brake']
        data['light'] = measurements[self.seq_len-1]['light_hazard']
        data['speed'] = measurements[self.seq_len-1]['speed']
        data['theta'] = measurements[self.seq_len-1]['theta']
        data['x_command'] = measurements[self.seq_len-1]['x_command']
        data['y_command'] = measurements[self.seq_len-1]['y_command']

        # target points
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        ego_theta = measurements[self.seq_len-1]['theta'] + rad # + rad for augmentation
        ego_x = measurements[self.seq_len-1]['x']
        ego_y = measurements[self.seq_len-1]['y']
        x_command = measurements[self.seq_len-1]['x_command']
        y_command = measurements[self.seq_len-1]['y_command']
        
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        local_command_point = R.T.dot(local_command_point)

        data['target_point'] = local_command_point
        
        data['target_point_image'] = draw_target_point(local_command_point)
        return data
