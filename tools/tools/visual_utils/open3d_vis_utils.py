"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import os
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, boxes, color, labels=None, scores=None):
    # for kitti
    #TODO
    box_colormap = [
        [1, 1, 1],  # unknown
        [1, 1, 0],  # yellow - ped
        [0, 1, 0],  # green - car
        [0, 1, 1],  # cyan - cyc
    ]

    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])

        if labels is not None:
            line_set.paint_uniform_color(box_colormap[labels[i]])
            vis.add_geometry(line_set)
        else:
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

        if scores is not None:
            continue
    return vis


def draw_sphere(vis, points, color, radius):
    num = points.shape[0]
    for i in range(num):
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        sphere.paint_uniform_color(color)
        sphere.translate(points[i])
        vis.add_geometry(sphere)
    return vis


def draw_line_between_points(vis, point1, point2, color, masks):
    assert (point1.shape == point2.shape)
    correnspondences = []

    point1 = point1[masks == 1]
    point2 = point2[masks == 1]

    for i in range(point1.shape[0]):
        correnspondences.append((i, i))

    p1 = open3d.geometry.PointCloud()
    p2 = open3d.geometry.PointCloud()

    p1.points = open3d.utility.Vector3dVector(point1)
    p2.points = open3d.utility.Vector3dVector(point2)

    line_set = open3d.geometry.LineSet.create_from_point_cloud_correspondences(p1, p2, correnspondences)
    line_set.points = open3d.utility.Vector3dVector(np.concatenate([point1, point2], axis=0))
    line_set.paint_uniform_color(color)
    vis.add_geometry(line_set)
    return vis


# visualize gt_boxes and pred_boxes
# only available batch_size=1
# kitti pass
def draw_scenes_v1(i, batch_dict, pred_dicts):
    points = batch_dict['points'][:, 1:] # raw points

    gt_boxes = batch_dict['gt_boxes'][0]
    pred_boxes = pred_dicts[0]['pred_boxes']

    gt_boxes_color = (0, 1, 0) # green
    pred_boxes_color = (1, 0, 0) # red

    camera_parameters_path = "./camera_parameters/camera_parameters_v1_"+str(i)+".json"
    has_camera_parameters = False
    if os.path.exists(camera_parameters_path):
        has_camera_parameters = True
        print("use exist camera parameters.")

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if isinstance(gt_boxes, torch.Tensor):
       gt_boxes = gt_boxes.cpu().numpy()

    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    # vis.create_window(window_name=str(i), width=1600, height=768)
    vis.create_window(window_name=str(i), width=800, height=600)

    if has_camera_parameters:
        ctr = vis.get_view_control()
        param = open3d.io.read_pinhole_camera_parameters(camera_parameters_path)

    vis.get_render_option().line_width = 1
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = (1, 1, 1) # black

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3))) # white
    vis.add_geometry(pts)

    if gt_boxes is not None:
        vis = draw_box(vis, boxes=gt_boxes, color=gt_boxes_color, labels=None, scores=None)

    if pred_boxes is not None:
        vis = draw_box(vis, boxes=pred_boxes, color=pred_boxes_color, labels=None, scores=None)

    if has_camera_parameters:
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()

    if not has_camera_parameters:
        if not os.path.exists("./camera_parameters"):
            os.makedirs("./camera_parameters")
            # print("make dir")
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(camera_parameters_path, param)
        print("save camera parameters.")

    vis.destroy_window()


# visualize vote centroids
# only available batch_size=1
# kitti pass
def draw_scenes_v2(i, batch_dict, pred_dicts):
    points = batch_dict['points'][:, 1:] # raw points
    centers = batch_dict['centers'] # centroids
    centers_origin = batch_dict['centers_origin']

    pred_boxes = pred_dicts[0]['pred_boxes']
    pred_scores = pred_dicts[0]['pred_scores']
    pred_labels = pred_dicts[0]['pred_labels']

    centers_color = (1, 0, 0) # red
    centers_origin_color = (1, 1, 0) # yellow
    pred_boxes_color = (1, 0, 0) # red

    camera_parameters_path = "./camera_parameters/camera_parameters_v2_"+str(i)+".json"
    has_camera_parameters = False
    if os.path.exists(camera_parameters_path):
        has_camera_parameters = True
        print("use exist camera parameters.")

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if isinstance(centers, torch.Tensor):
        centers = centers.cpu().numpy()

    if isinstance(centers_origin, torch.Tensor):
        centers_origin = centers_origin.cpu().numpy()

    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=str(i), width=1600, height=768)
    # vis.create_window(window_name=str(i), width=800, height=600)

    if has_camera_parameters:
        ctr = vis.get_view_control()
        param = open3d.io.read_pinhole_camera_parameters(camera_parameters_path)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = (0, 0, 0) # black

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3))) # white
    vis.add_geometry(pts)

    if pred_boxes is not None:
        vis = draw_box(vis, boxes=pred_boxes, color=pred_boxes_color, labels=None, scores=None)

    if centers is not None:
        vis = draw_sphere(vis, points=centers[:, 1:4], color=centers_color, radius=0.1)

    if centers_origin is not None:
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(centers[:, 1:4], pred_boxes[:, :7])
        point_masks = point_masks.sum(axis=0)
        vis = draw_sphere(vis, points=centers_origin[:, 1:4], color=centers_origin_color, radius=0.1)
        vis = draw_line_between_points(vis, point1=centers_origin[:, 1:4], point2=centers[:, 1:4], color=centers_origin_color, masks=point_masks)

    if has_camera_parameters:
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()

    if not has_camera_parameters:
        if not os.path.exists("./camera_parameters"):
            os.makedirs("./camera_parameters")
            # print("make dir")
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(camera_parameters_path, param)
        print("save camera parameters.")

    vis.destroy_window()


def draw_scenes_vote_points_has_label(points, vote_points, origin_centers,
                                      gt_boxes=None, ref_boxes=None, ref_labels=None,
                                      ref_scores=None, point_colors=None, vote_point_colors = None,
                                      draw_origin=True ):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(vote_points, torch.Tensor):
        vote_points = vote_points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(origin_centers, torch.Tensor):
        origin_centers = origin_centers.cpu().numpy()

    # 初始化
    app = open3d.visualization.gui.Application.instance
    app.initialize()
    vis = open3d.visualization.O3DVisualizer("open3d_text")
    vis.show_settings = True
    vis.show_skybox(False)
    vis.set_background([0, 0, 0, 0], None)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    # 整体点云染色
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    # gt
    # if gt_boxes is not None:
    #     vis = draw_box(vis, gt_boxes, (0, 1, 0))
    # ref
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), None, ref_scores)
    #vote line   只保留中心点在gt内
    if origin_centers is not None:
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(vote_points[:, 1:4], ref_boxes[:, :7])
        point_masks = point_masks.sum(axis=0)
        vis = draw_vote_line(vis, origin_centers[:, 1:4], vote_points[:, 1:4], masks=point_masks)

    #添加图形到vis
    vis.add_geometry("points", pts)
    draw_point_as_sphere(vis, origin_centers[:, 1:4], radius=0.1, color=(1, 1, 0), name_tag=0)
    draw_point_as_sphere(vis, vote_points[:, 1:4], radius=0.1, color=(1, 0, 0), name_tag=1)
    #vis.add_geometry(origin_pts)

    app.add_window(vis)
    app.run()

def draw_scenes_vote_points_has_label_v2(points, vote_points, origin_centers, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, vote_point_colors = None, draw_origin=True ):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(vote_points, torch.Tensor):
        vote_points = vote_points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(origin_centers, torch.Tensor):
        origin_centers = origin_centers.cpu().numpy()

    # 初始化
    app = gui.Application.instance
    app.initialize()
    vis = open3d.visualization.O3DVisualizer("open3d_text")
    vis.show_settings = True
    vis.show_skybox(False)
    vis.set_background([0, 0, 0, 0], None)
    vis.enable_raw_mode(True)
    # ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters("/home/hdwu/qkl/CA-SSD/tools/visual_result/ScreenCamera_2023-05-22-16-07-56.json")



    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    # vis.point_size(0.3)

    # 整体点云染色
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    # gt
    if gt_boxes is not None:
        vis = draw_box(vis,  gt_boxes, (0, 1, 0))
    # ref
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 0, 1), ref_labels, ref_scores)
    #vote line   只保留中心点在gt内
    if origin_centers is not None:
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(vote_points[:, 1:4], gt_boxes[:, :7])
        point_masks = point_masks.sum(axis=0)
        vis = draw_vote_line(vis, origin_centers[:, 1:4], vote_points[:, 1:4], masks=point_masks)

    #添加图形到vis
    vis.add_geometry("points", pts)
    draw_point_as_sphere(vis, origin_centers[:, 1:4], radius=0.1, color=(1, 1, 0), name_tag=0)
    draw_point_as_sphere(vis, vote_points[:, 1:4], radius=0.1, color=(1, 0, 0), name_tag=1)
    #vis.add_geometry(origin_pts)
    vis.point_size = 1
    vis.reset_camera_to_default()
    vis.setup_camera(param.intrinsic, param.extrinsic)
    app.add_window(vis)
    app.run()

    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # open3d.io.write_pinhole_camera_parameters("/home/limo/PCD/IA-SSD-main/tools/visual_utils/test_2.json", param)


def draw_vote_line(vis, point1, point2, masks):
    assert (point1.shape == point2.shape)
    correnspondences = []
    point1 = point1[masks == 1]
    point2 = point2[masks == 1]
    for i in range(point1.shape[0]):
        correnspondences.append((i, i))

    p1 = p2 = open3d.geometry.PointCloud()

    p1.points = open3d.utility.Vector3dVector(point1)
    p2.points = open3d.utility.Vector3dVector(point2)
    line_set = open3d.geometry.LineSet.create_from_point_cloud_correspondences(p1, p2, correnspondences)
    line_set.points = open3d.utility.Vector3dVector(np.concatenate([point1, point2], axis=0))
    line_set.paint_uniform_color((1, 1, 0))  # RGB
    # vis.add_geometry("vote_line", line_set)
    vis.add_geometry(line_set)
    return vis



def draw_point_as_sphere(vis, points, color=(1, 0, 0), radius=1.0, name_tag = 0):
    num = points.shape[0]
    for i in range(num):
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = 'defaultLit'
        sphere.translate(points[i])
        # vis.add_geometry("{}sphere{}".format(name_tag, i), sphere)
        vis.add_geometry(sphere)
    return vis
