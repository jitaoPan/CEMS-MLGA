import numpy as np
import torch.nn as nn
import torch 
from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict



class AnchorHeadSingleWithMLGA(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1
        )
        self.mlga_enabled = model_cfg.get('MLGA', {}).get('ENABLED', 0)
        if self.mlga_enabled:
            self._build_mlga_module({}, input_channels)
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None):
            self.conv_dir_cls = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    def _build_mlga_module(self, mlga_cfg, input_channels):
        self.offset_mlp = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(),
            nn.Linear(input_channels // 2, 3)
        )
        self.k1 = mlga_cfg.get('K1', 16)
        self.k2 = mlga_cfg.get('K2', 8)
        self.m = mlga_cfg.get('M', 4)
        self.query = nn.Linear(input_channels, input_channels)
        self.key = nn.Linear(input_channels, input_channels)
        self.value = nn.Linear(input_channels, input_channels)
        self.ffn = nn.Sequential(
            nn.Linear(input_channels * 2, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, input_channels)
        )
        self.geo_head = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(),
            nn.Linear(input_channels // 2, 7)
        )
        self.tau1 = mlga_cfg.get('TAU1', 0.3)
        self.tau2 = mlga_cfg.get('TAU2', 0.2)
    def _mlga_forward(self, spatial_features_2d, batch_dict):
        B, C, H, W = spatial_features_2d.shape
        cls_scores = self.conv_cls(spatial_features_2d).sigmoid()
        max_scores, _ = cls_scores.max(dim=1)
        flat_scores = max_scores.view(B, -1)
        topk_values, topk_indices = torch.topk(flat_scores, k=1024, dim=1)
        mask = topk_values > self.tau1
        selected_indices = topk_indices[mask]
        if selected_indices.numel() == 0:
            return spatial_features_2d, 0.0
        selected_coords = torch.stack([
            selected_indices // W,
            selected_indices % W
        ], dim=1).float()
        selected_features = spatial_features_2d.permute(0, 2, 3, 1).reshape(B, -1, C)[
            torch.arange(B).unsqueeze(1), selected_indices // (H * W)
        ]
        offsets = self.offset_mlp(selected_features)
        updated_coords = selected_coords + offsets[:, :2]
        dist_matrix = torch.cdist(updated_coords, updated_coords)
        _, g1_indices = torch.topk(dist_matrix, k=self.k1, dim=1, largest=False)
        g2_indices = []
        for i in range(g1_indices.shape[0]):
            _, local_indices = torch.topk(dist_matrix[i, g1_indices[i]], k=self.m, largest=False)
            g2_indices.append(g1_indices[i, local_indices])
        g2_indices = torch.stack(g2_indices)
        g3_features = selected_features.unsqueeze(1).repeat(1, self.k2, 1)
        q = self.query(selected_features).unsqueeze(1)
        k = self.key(selected_features[g1_indices])
        v = self.value(selected_features[g1_indices])
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(C)
        attn_weights = F.softmax(attn_scores, dim=-1)
        g1_out = torch.bmm(attn_weights, v).squeeze(1)
        q_cross = self.query(selected_features).unsqueeze(1)
        k_cross = self.key(g3_features)
        cross_attn = torch.bmm(q_cross, k_cross.transpose(1, 2)) / np.sqrt(C)
        cross_weights = F.softmax(cross_attn, dim=-1)
        g3_out = torch.bmm(cross_weights, g3_features).squeeze(1)
        fused_features = self.ffn(torch.cat([g1_out, g3_out], dim=1))
        enhanced_features = selected_features + fused_features
        if self.training:
            gt_boxes = batch_dict['gt_boxes']
            pred_boxes = self._decode_boxes(enhanced_features, updated_coords)
            iou_matrix = self._calculate_iou(pred_boxes, gt_boxes)
            max_iou, _ = iou_matrix.max(dim=2)
            keep_mask = max_iou > 0.1
            enhanced_features = enhanced_features[keep_mask]
        geo_pred = self.geo_head(enhanced_features)
        geo_loss = self._compute_geo_loss(geo_pred, batch_dict)
        output_features = spatial_features_2d.clone()
        for b in range(B):
            batch_mask = (selected_indices // (H * W)) == b
            if batch_mask.any():
                output_features[b].view(C, -1)[:, selected_indices[batch_mask]] = \
                    enhanced_features[batch_mask].t()

        return output_features, geo_loss
    def _decode_boxes(self, features, coords):
        box_params = self.geo_head(features)
        boxes = torch.cat([
            coords, box_params
        ], dim=1)
        return boxes
    def _calculate_iou(self, boxes1, boxes2):
        return torch.min(boxes1[:, None, 3:6], boxes2[:, 3:6]).prod(dim=2) / \
            torch.max(boxes1[:, None, 3:6], boxes2[:, 3:6]).prod(dim=2)
    def _compute_geo_loss(self, pred, batch_dict):
        if not self.training or 'gt_boxes' not in batch_dict:
            return 0.0
        gt_boxes = batch_dict['gt_boxes']
        gt_sizes = gt_boxes[..., 3:6]
        gt_angles = gt_boxes[..., 6]
        pred_sizes = pred[..., :3]
        pred_angles = pred[..., 3]
        size_loss = F.smooth_l1_loss(pred_sizes, gt_sizes[:, 0], reduction='mean')
        angle_loss = F.smooth_l1_loss(pred_angles, gt_angles[:, 0], reduction='mean')
        return size_loss + angle_loss

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        if self.mlga_enabled:
            spatial_features_2d, geo_loss = self._mlga_forward(spatial_features_2d, data_dict)
            if self.training:
                data_dict['geo_loss'] = geo_loss
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        return data_dict