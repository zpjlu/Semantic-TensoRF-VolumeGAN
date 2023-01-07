log=0101_1
mkdir bk/${log}
cp train.py bk/${log}/
cp models/semantic_stylegan.py bk/${log}/
CUDA_VISIBLE_DEVICES=0 python -u train.py \
--dataset data/lmdb_celebamaskhq_512 \
--inception data/inception_celebamaskhq_512.pkl \
--checkpoint_dir test/${log} \
--seg_dim 13 \
--size 512 \
--transparent_dims 10 12 \
--residual_refine \
--batch 4 \
--num_pre_train 1000 \
--sample_model 1 \
| tee ${log}.log