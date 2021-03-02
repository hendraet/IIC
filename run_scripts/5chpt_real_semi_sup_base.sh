MODEL_IND=1003
DATASET_NAME="5CHPT_plus_unlabelled"
EXP_NAME="5CHPT_real_semi_sup_base"

mkdir -p out/$MODEL_IND
export CUDA_VISIBLE_DEVICES=$1 &&
python2 \
    -m src.scripts.cluster.cluster \
    $MODEL_IND \
    /home.new/hendrik/datasets/ \
    $DATASET_NAME \
    --batch_sz 128 \
    --lr 0.001 \
    --arch ClusterNet5g \
    --sobel \
    --batchnorm_track \
    --mode IID+ \
    --gt_k 5 \
    --output_ks 35 \
    --num_dataloaders 4 \
    --num_sub_heads 5 \
    --crop_orig \
    --rand_crop_sz 1.0 \
    --out_root out \
    --mix_train \
    --input_sz 64 216 \
    > out/$MODEL_IND/$EXP_NAME.out &
