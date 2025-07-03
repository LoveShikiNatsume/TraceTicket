echo "Usage: bash train.sh [dataset_path]"
echo "DATASET: $1"
# rm -r results
python3 -m tracegnn.models.trace_vae.train --device=cpu --dataset.root_dir="$1" --seed=1234 --model.struct.z_dim=10 --model.struct.decoder.use_prior_flow=true --train.z_unit_ball_reg=1 --model.latency.z2_dim=10 --model.latency.decoder.condition_on_z=true --output-dir="results/train/"
# python3 -m tracegnn.models.trace_vae.train --device=cpu --dataset.root_dir="$1" --seed=1234 --model.struct.z_dim=10 --model.struct.decoder.use_prior_flow=true --train.z_unit_ball_reg=1 --model.latency.z2_dim=10 --model.latency.decoder.condition_on_z=true --output-dir="results/train/" --resume=never

# 只调整关键参数，降低风险
# python3 -m tracegnn.models.trace_vae.train \
#     --device=cpu \
#     --dataset.root_dir="/home/fuxian/tracevae/TT_Dataset/TT_Dataset/convert_data_time_corrected" \
#     --seed=1234 \
#     --model.operation_embedding_dim=50 \
#     --model.struct.z_dim=15 \
#     --model.struct.decoder.use_prior_flow=true \
#     --model.latency.z2_dim=15 \
#     --model.latency.decoder.condition_on_z=true \
#     --model.latency.decoder.biased_normal_std_threshold=3.0 \
#     --train.z_unit_ball_reg=1 \
#     --output-dir="results/train_optimize/"