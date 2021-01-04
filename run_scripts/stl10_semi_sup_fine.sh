MODEL_IND=698
OLD_MODEL_IND=650
EXP_NAME="STL10_semi_sup_fine"

mkdir -p out/$MODEL_IND
export CUDA_VISIBLE_DEVICES=$1 &&
python \
    -m src.scripts.semisup.IID_semisup \
    $MODEL_IND \
    --old_model_ind $OLD_MODEL_IND \
    --head_lr 0.001 \
    --trunk_lr 0.0001 \
    --arch SupHead5 \
    --penultimate_features \
    --random_affine \
    --affine_p 0.5 \
    --cutout \
    --cutout_p 0.5 \
    --cutout_max_box 0.7 \
    --num_epochs 8000 \
    --out_root out \
    > out/$MODEL_IND/$EXP_NAME.out &
