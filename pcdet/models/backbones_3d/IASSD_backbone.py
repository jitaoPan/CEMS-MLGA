import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os


class IASSD_Backbone(nn.Module):
    """ Backbone for IA-SSD"""

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)

        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list):  ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                    channel_out += self.num_class
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class=self.num_class
                    )
                )

            elif self.layer_types[k] == 'UP_Layer':
                self.SA_modules.append(pointnet2_modules.UpSampling_layer(pre_channel=channel_out_list[k]+channel_out_list[k-1],
                                                                          mlp_list=sa_config.MLPS[k],
                                                                          confidence_mlp=self.confidence_mlps[k],
                                                                          num_class= self.num_class))
                channel_out = self.aggregation_mlps[k][0]
                channel_out += self.num_class

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]]+self.num_class,
                                                                    max_translate_range=self.max_translate_range
                                                                    )
                                       )

            elif self.layer_types[k] == 'Attn_Layer':
                in_dim = channel_out_list[self.layer_inputs[k]]                
                mlp_config = sa_config.MLPS[k]
                attention_dim = in_dim
                # 修改参数
                k1 = 16 
                k2 = 3 
                m = 16                
                self.SA_modules.append(
                    pointnet2_modules.MultiLevelGraphAttention(
                        in_dim=in_dim,
                        attention_dim=attention_dim,
                        k1=k1,
                        k2=k2,
                        m=m,    
                        num_class=self.num_class,
                        aggregation_mlp=self.aggregation_mlps[k] if self.aggregation_mlps else None,
                        confidence_mlp=self.confidence_mlps[k] if self.confidence_mlps else None
                    )
                )
                channel_out = 1024
            channel_out_list.append(channel_out)

        self.num_point_features = channel_out

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def suppress_non_essential_keypoints(self, centers, centers_features, cls_preds, batch_idx):
        if not self.suppress_keypoints:
            return centers, centers_features, cls_preds, batch_idx

        batch_size = batch_idx.max().item() + 1
        filtered_centers = []
        filtered_features = []
        filtered_cls_preds = []
        filtered_batch_idx = []

        for bs_idx in range(batch_size):
            mask = (batch_idx == bs_idx)
            bs_centers = centers[mask]
            bs_features = centers_features[mask]
            bs_cls_preds = cls_preds[mask]

            cls_scores, cls_labels = torch.max(bs_cls_preds, dim=1)

            sorted_indices = torch.argsort(cls_scores, descending=True)
            remaining_indices = sorted_indices.clone().tolist()
            final_indices = []

            while len(remaining_indices) > 0:
                current_idx = remaining_indices.pop(0)
                final_indices.append(current_idx)

                if len(remaining_indices) == 0:
                    break

                current_class = cls_labels[current_idx].item()
                current_pos = bs_centers[current_idx]
                radius = self.class_radius.get(current_class, 0.6)

                distances = torch.norm(bs_centers[remaining_indices] - current_pos, dim=1)

                same_class_mask = cls_labels[remaining_indices] == current_class
                within_radius_mask = distances < radius
                remove_mask = same_class_mask & within_radius_mask

                remove_indices = torch.where(remove_mask)[0]
                for idx in sorted(remove_indices.tolist(), reverse=True):
                    del remaining_indices[idx]

            filtered_centers.append(bs_centers[final_indices])
            filtered_features.append(bs_features[final_indices])
            filtered_cls_preds.append(bs_cls_preds[final_indices])
            filtered_batch_idx.append(torch.full((len(final_indices),), bs_idx,
                                                 dtype=torch.long, device=batch_idx.device))

        return (torch.cat(filtered_centers, dim=0),
                torch.cat(filtered_features, dim=0),
                torch.cat(filtered_cls_preds, dim=0),
                torch.cat(filtered_batch_idx, dim=0))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2,
                                                                             1).contiguous() if features is not None else None  ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred,
                                                                      ctr_xyz=ctr_xyz)

            elif self.layer_types[i] == 'UP_Layer':
                old_xyz_input = encoder_xyz[self.layer_inputs[i-1]]
                old_feature_input = encoder_features[self.layer_inputs[i-1]]
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](old_xyz_input, xyz_input, old_feature_input, feature_input)

            elif self.layer_types[i] == 'Vote_Layer':  # i=4
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                encoder_coords.append(
                    torch.cat([center_origin_batch_idx[..., None].float(), centers_origin.view(batch_size, -1, 3)],
                              dim=-1))

            elif self.layer_types[i] == 'Attn_Layer':
                prev_level_features = encoder_features[self.layer_inputs[i]-1] if self.layer_inputs[i] > 0 else None
                prev_level_xyz = encoder_xyz[self.layer_inputs[i]-1] if self.layer_inputs[i] > 0 else None
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](
                    xyz_input, 
                    feature_input,
                    prev_level_xyz=prev_level_xyz,
                    prev_level_features=prev_level_features
                )

            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(), li_xyz.view(batch_size, -1, 3)], dim=-1))
            encoder_features.append(li_features)
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat(
                    [li_cls_batch_idx[..., None].float(), li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],
                    dim=-1))
            else:
                sa_ins_preds.append([])

        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)),
                                              dim=1)

        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat(
            (ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)

        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx
        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features
        # if not self.training:
        #     # Apply non-essential key point suppression
        #     centers_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        #     cls_preds = sa_ins_preds[-1][..., 1:] if len(sa_ins_preds[-1]) > 0 else None
        #     if cls_preds is not None and self.suppress_keypoints:
        #         centers, centers_features, cls_preds, ctr_batch_idx = self.suppress_non_essential_keypoints(
        #             centers.view(-1, 3), centers_features, cls_preds.view(-1, cls_preds.shape[-1]), ctr_batch_idx
        #         )
        #
        #         # Update batch_dict with filtered results
        #         batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers), dim=1)
        #         batch_dict['centers_features'] = centers_features
        #         batch_dict['ctr_batch_idx'] = ctr_batch_idx

        # save per frame
        if self.model_cfg.SA_CONFIG.get('SAVE_SAMPLE_LIST', False) and not self.training:
            import numpy as np
            result_dir = '/home/hdwu/qkl/CA-SSD/sample_list_save/'
            for i in range(batch_size):
                idx = batch_dict['frame_id'][i]
                xyz_list = []
                for sa_xyz in encoder_xyz:
                    xyz_list.append(sa_xyz[i].cpu().numpy())

                sample_xyz = result_dir + idx + ".npy"
                np.save(sample_xyz, xyz_list)

        return batch_dict