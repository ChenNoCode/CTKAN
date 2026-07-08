dataset=busi
input_size=256
python train.py --arch CTKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name CTKAN --resume --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}/CTKAN

dataset=glas
input_size=512
python train.py --arch CTKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name CTKAN --resume --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}/CTKAN

dataset=cvc
input_size=256
python train.py --arch CTKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name CTKAN --resume --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}/CTKAN

