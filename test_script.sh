cd /workspace/MICCAI23-ProtoContra-SFDA

python3 test.py \
    --model_path '/workspace/MICCAI23-ProtoContra-SFDA/results/Target_Adapt/UNet_Abdomen_MR2CT_Adapt_CL/exp_1_time_2026-03-27 08:19:40/saved_models/best_model_step_410_dice_0.7130.pth' \
    --data_root datasets/chaos \
    --target_site CT \
    --gpu_id 0