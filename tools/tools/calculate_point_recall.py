import pickle
import numpy as np
import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


def name_to_cls_once(gt_boxes_name):
    gt_boxes_cls = np.zeros(gt_boxes_name.shape[0], dtype=int)

    # ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
    car_list = ['Car', 'Bus', 'Truck']
    ped_list = ['Pedestrian']
    cyc_list = ['Cyclist']

    for i in range(gt_boxes_name.shape[0]):
        if gt_boxes_name[i] in car_list:
            gt_boxes_cls[i] = 1
            continue

        if gt_boxes_name[i] in ped_list:
            gt_boxes_cls[i] = 2
            continue

        if gt_boxes_name[i] in cyc_list:
            gt_boxes_cls[i] = 3
    return gt_boxes_cls

if __name__ == '__main__':
    sample_data_dir = "/home/hdwu/qkl/CA-SSD/sample_once_cams/"
    pkl_file = '/home/hdwu/qkl/CA-SSD/data/once/val_3321.pkl'
    with open(pkl_file, 'rb') as f:
        pkl_infos = pickle.load(f)

    gt_cls_num = [0, 0, 0]
    sample_cls_num = [0, 0, 0]

    for info in pkl_infos:
        lidar_idx = info['frame_id']
        sample_file_path = sample_data_dir + lidar_idx + '.npy'
        sample_file = np.load(sample_file_path, allow_pickle=True)
        points = sample_file[1]
        gt_boxes_lidar = info['annos']['boxes_3d']
        gt_boxes_name = info['annos']['name']
        gt_boxes_cls = name_to_cls_once(gt_boxes_name)

        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes_lidar)
        ).numpy()  # (nboxes, npoints)

        # gt_points = points[point_indices[i] > 0]
        for i in range(point_indices.shape[0]):
            points_idx = point_indices[i]
            cls = gt_boxes_cls[i]
            gt_cls_num[cls-1] += 1
            in_gt = np.sum(points_idx) != 0
            if in_gt:
                sample_cls_num[cls-1] += 1

        # draw_scenes(
        #     points=points, gt_boxes=gt_boxes_lidar, gt_boxes_cls=gt_boxes_cls
        # )
    print(points.shape)
    print(gt_cls_num)
    print(sample_cls_num)
    print("car: %.4f" % (sample_cls_num[0] / gt_cls_num[0]))
    print("ped: %.4f" % (sample_cls_num[1] / gt_cls_num[1]))
    print("cyc: %.4f" % (sample_cls_num[2] / gt_cls_num[2]))