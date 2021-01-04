MODEL_IND=1000
export CUDA_VISIBLE_DEVICES=$1 &&
python \
    -m src.scripts.cluster.cluster \
    $MODEL_IND \
    datasets/ \
    5CHPT \
    --batch_sz 256 \
    --lr 0.0001 \
    --arch ClusterNet6cTwoHead \
    --batchnorm_track \
    --mode IID \
    --gt_k 5 \
    --output_ks 25 5 \
    --num_dataloaders 4 \
    --num_sub_heads 5 \
    --crop_orig \
    --crop_other \
    --tf1_crop centre_half \
    --tf2_crop random \
    --rot_val 25 \
    --no_flip \
    --head_epochs 1 2 \
    --out_root out \
    > out/$MODEL_IND/5CHPT.out &

