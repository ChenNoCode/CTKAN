source /root/miniconda3/etc/profile.d/conda.sh
conda activate ctkan
cd /home/code/CTKAN

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0

model=CTKAN
batch_size=4
seed=42
dataset=glas
input_size=512

python train.py --arch ${model} --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${model} --epochs 400 --batch_size ${batch_size} --seed ${seed} --data_dir ./inputs
python val.py --name ${dataset}/${model}


