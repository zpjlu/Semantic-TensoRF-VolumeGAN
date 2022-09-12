python -u train.py \
--dataset data/lmdb_celebamaskhq_512 \
--inception data/inception_celebamaskhq_512.pkl \
--checkpoint_dir checkpoint/celebamaskhq_512 \
--seg_dim 13 \
--size 512 \
--transparent_dims 10 12 \
--residual_refine \
--batch 4 