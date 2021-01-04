MODEL_IND=569
DATASET_NAME="STL10"
EXP_NAME="stl10_w_double_eval"

export CUDA_VISIBLE_DEVICES=$1 &&
python \
    -m src.scripts.cluster.cluster_sobel_twohead \
    --model_ind $MODEL_IND \
    --arch ClusterNet5gTwoHead \
    --mode IID \
    --dataset $DATASET_NAME \
    --dataset_root datasets \
    --gt_k 10 \
    --output_k_A 70 \
    --output_k_B 10 \
    --lamb 1.0 \
    --lr 0.0001  \
    --num_epochs 400 \
    --batch_sz 350 \
    --num_dataloaders 5 \
    --num_sub_heads 5 \
    --mix_train \
    --crop_orig \
    --rand_crop_sz 0.9 \
    --input_sz 64 \
    --head_A_first \
    --double_eval \
    --batchnorm_track \
    --out_root out/ \
    > out/$MODEL_IND/$EXP_NAME.out &
