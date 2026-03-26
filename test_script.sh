cd /opt/data/private/MICCAI23-ProtoContra-SFDA

python3 test.py \
    --model_path '/opt/data/private/MICCAI23-ProtoContra-SFDA/results/Target_Adapt/UNet_Abdomen_CT2MR_Adapt_CL/exp_2_time_2026-03-26 13:06:48/saved_models/model_step_20_dice_0.8495.pth' \
    --data_root datasets/chaos \
    --target_site MR \
    --gpu_id 0