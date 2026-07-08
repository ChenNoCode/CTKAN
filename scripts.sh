export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0

model=CTKAN
batch_size=4
seed=42

dataset=busi
input_size=256
python train.py --arch ${model} --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${model} --epochs 400 --batch_size ${batch_size} --seed ${seed} --data_dir ./inputs
python val.py --name ${dataset}/${model}

# dataset=heus
# input_size=256
# python train.py --arch ${model} --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${model} --epochs 400 --batch_size ${batch_size} --seed ${seed} --data_dir ./inputs
# python val.py --name ${dataset}/${model}

# dataset=cvc
# input_size=256
# python train.py --arch ${model} --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${model} --epochs 400 --batch_size ${batch_size} --seed ${seed} --data_dir ./inputs
# python val.py --name ${dataset}/${model}

