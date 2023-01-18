import random
from copy import deepcopy

import cv2
import numpy as np
from skimage.transform import rotate


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T
        
def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_lidar_to_vehicle_transform():
    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=np.float32)
    T = np.eye(4)
    T[:3, :3] = rot

    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def get_vehicle_to_lidar_transform():
    return np.linalg.inv(get_lidar_to_vehicle_transform())

def get_lidar_to_bevimage_transform():
    # rot 
    T = np.array([[0, -1, 16],
                  [-1, 0, 32],
                  [0, 0, 1]], dtype=np.float32)
    # scale 
    T[:2, :] *= 8

    return T


def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized
    #clip to 50 meters
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 # Rescale map to lie in [0,1]

    return normalized


def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

# this is only for visualization, For training, we should use vehicle coordinate

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    T = get_vehicle_to_virtual_lidar_transform()
    
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
            
    return waypoints

def align(lidar_0, measurements_0, measurements_1, degree=0):
    
    matrix_0 = measurements_0['ego_matrix']
    matrix_1 = measurements_1['ego_matrix']

    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)
   
    Tr_lidar_to_vehicle = get_lidar_to_vehicle_transform()
    Tr_vehicle_to_lidar = get_vehicle_to_lidar_transform()

    transform_0_to_1 = Tr_vehicle_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_vehicle

    # augmentation
    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0],
                              [-np.sin(rad), np.cos(rad), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1
                            
    lidar = lidar_0.copy()
    lidar[:, -1] = 1.
    #important we should convert the points back to carla format because when we save the data we negatived y component
    # and now we change it back 
    lidar[:, 1] *= -1.
    lidar = transform_0_to_1 @ lidar.T
    lidar = lidar.T
    lidar[:, -1] = lidar_0[:, -1]
    # and we change back here
    lidar[:, 1] *= -1.

    return lidar


def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def get_bbox_label(bbox, rad=0):
    # dx, dy, dz, x, y, z, yaw
    # ignore z
    dz, dx, dy, x, y, z, yaw, speed, brake =  bbox

    pixels_per_meter = 8

    # augmentation
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                              [-np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position

    position = np.clip(position, 0., 255.)
    x, y = position[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, dy*pixels_per_meter, dx*pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox


def parse_labels(labels, rad=0):
    bboxes = {}
    for result in labels:
        num_points = result['num_points']
        distance = result['distance']

        x = result['position'][0]
        y = result['position'][1]

        bbox = result['extent'] + result['position'] + [result['yaw'], result['speed'], result['brake']]
        bbox = get_bbox_label(bbox, rad)

        # Filter bb that are outside of the LiDAR after the random augmentation. The bounding box is now in image space
        if num_points <= 1 or bbox[0] <= 0.0 or bbox[0] >= 255.0 or bbox[1] <= 0.0 or bbox[1] >=255.0:
            continue

        bboxes[result['id']] = bbox
    return bboxes

def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized

def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized

def crop_image(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    
    # only shift for x direction
    start_x += int(crop_shift)

    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def scale_seg(image, scale):
    (width, height) = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    if scale != 1:
        im_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        im_resized = image
    return im_resized

def crop_seg(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a seg image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop

    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cropped_image

def load_crop_bev_npy(bev_array, degree):
    """
    Load and crop an Image.
    Crop depends on augmentation angle.
    """
    PIXELS_PER_METER_FOR_BEV = 5
    PIXLES = 32 * PIXELS_PER_METER_FOR_BEV
    start_x = 250 - PIXLES // 2
    start_y = 250 - PIXLES

    # shift the center by 7 because the lidar is + 1.3 in x 
    bev_array = np.moveaxis(bev_array, 0, -1).astype(np.float32)
    bev_shift = np.zeros_like(bev_array)
    bev_shift[7:] = bev_array[:-7]

    bev_shift = rotate(bev_shift, degree)
    cropped_image = bev_shift[start_y:start_y+PIXLES, start_x:start_x+PIXLES]
    cropped_image = np.moveaxis(cropped_image, -1, 0)

    # we need to predict others so append 0 to the first channel
    cropped_image = np.concatenate((np.zeros_like(cropped_image[:1]), 
                                    cropped_image[:1],
                                    cropped_image[:1] + cropped_image[1:2]), axis=0)

    cropped_image = np.argmax(cropped_image, axis=0)
    
    return cropped_image



def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[1] += 1.3
    point = target_point * 8.
    point[1] *= -1
    point[1] = 256 - point[1] 
    point[0] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float) / 255.

def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    """
    Compute projections between LiDAR BEV and image space
    """
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))

    tmp_bev = np.empty((lidar_x, lidar_y, ), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y, ), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i,j] = []

    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []

    for i in range(valid_bev_points.shape[0]):
        tmp_bev[valid_bev_points[i][0]//scale, valid_bev_points[i][1]//scale].append(valid_cam_points[i]//scale)
        tmp_cam[valid_cam_points[i][0]//scale, valid_cam_points[i][1]//scale].append(valid_bev_points[i]//scale)

    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i,j]

            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i,j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i,j,:num_points] = np.array(cam_to_bev_points)

    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i,j]

            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i,j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i,j,:num_points] = np.array(bev_to_cam_points)

    return cam_to_bev_proj_locs, bev_to_cam_proj_locs

def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from CARLA in the CARLA coordinate system: x left, y forward, z up (LiDAR rotated by 90 degree)
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """

    pixels_per_meter = 8
    lidar_width      = 256
    lidar_height     = 256
    lidar_meters_x   = (lidar_width  / pixels_per_meter) / 2 # Divided by two because the LiDAR is in the center of the image
    lidar_meters_y   =  lidar_height / pixels_per_meter

    downscale_factor = 32

    img_width  = 352
    img_height = 160
    fov_width  = 60

    left_camera_rotation  = -60.0
    right_camera_rotation =  60.0

    fov_height = 2.0 * np.arctan((img_height / img_width) * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)

    # Our pixels are squares so focal_x = focal_y
    focal_x = img_width  / (2.0 * np.tan(np.deg2rad(fov_width)  / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))

    cam_z   = 2.3
    lidar_z = 2.5

    # get valid points in 64x64 grid
    world[:, 0] *= -1  # flip x axis, so that the positive direction points towards right. new coordinate system: x right, y forward, z up
    lidar = world[abs(world[:,0])<lidar_meters_x] # 32m to the sides
    lidar = lidar[lidar[:,1]<lidar_meters_y] # 64m to the front
    lidar = lidar[lidar[:,1]>0] # 0m to the back

    # Translate Lidar cloud to the same coordinate system as the cameras (They only differ in height)
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)

    # Make copies because we will rotate the new pointclouds
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)


    lidar_indices = np.arange(0, lidar.shape[0], 1)
    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar[..., 1]
    x = ((focal_x * lidar[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar[..., 2]) / z) + (img_height / 2.0)
    result_center = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
    result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]

    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)

    # Rotate the left camera to align with the axis for projection with a pinhole camera model
    theta = np.radians(left_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_left_camera[..., 1]
    x = ((focal_x * lidar_for_left_camera[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar_for_left_camera[..., 2]) / z) + (img_height / 2.0)
    result_left = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
    result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
    result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)

    # Do the same for the right image
    theta = np.radians(right_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_right_camera[..., 1]
    x = ((focal_x * lidar_for_right_camera[..., 0]) / z) + (img_width / 2.0)
    y = ((focal_y * lidar_for_right_camera[..., 2]) / z) + (img_height / 2.0)
    result_right = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] # Cut of right part, it's not used.
    result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width

    # Combine the three images into one
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    if(debug == True):
        # Visualize LiDAR hits in image
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        vis_original_image = image_vis[0].detach().cpu().numpy()
        vis_original_image = np.transpose(vis_original_image, (1, 2, 0)) / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        lidar_vis = lidar_vis.detach().cpu().numpy()
        vis_original_lidar[np.greater(lidar_vis[0,0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[0,1], 0)] = 255


    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        # Project the LiDAR point to BEV and save index of the BEV image pixel.
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height-1)) * -1

        valid_bev_points.append([bev_x, bev_y])
        # Calculate index in the final image by rounding down
        img_x = int(results_total[i][0])
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])


        if (debug == True):
            vis_original_image[img_y, img_x] = np.array([0.0,1.0,0.0])
            vis_bev[bev_y, bev_x] = 255 #Debug visualization
            vis[img_y, img_x] = 255

    if (debug == True):
        # NOTE add the paths you want the images to land in here before debugging
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()


    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)

    return bev_points, cam_points

def decode_pil_to_npy(img):
    """
    """
    (channels, width, height) = (15, img.shape[1], img.shape[2])

    bev_array = np.zeros([channels, width, height])

    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos

    # hard coded to select
    return bev_array[10:12]