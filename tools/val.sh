extra_tag=12-8-hiera_sampling_attn-3

for thresh in 0.1 0.3 0.5 0.7
do
for batch_size in 4 8 12 16 20
do
python test.py --cfg_file cfgs/kitti_models/IA-SSD.yaml --eval_all --ckpt_dir ../output/kitti_models/IA-SSD/$extra_tag/ckpt/ --eval_tag bs$batch_size-nms$thresh --extra_tag $extra_tag --batch_size $batch_size --set MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH $thresh
done
done
