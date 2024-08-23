CUDA_LAUNCH_BLOCKING=1 python train_textBPN.py --exp_name MySynthData --net resnet50 --scale 1 --max_epoch 60 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.001 --num_workers 4

python train_textBPN.py --exp_name MySynthData --net resnet50 --scale 1 --max_epoch 300 --batch_size 12 --gpu 0 --input_size 640 --optim Adam --lr 0.0001 --num_workers 4 --load_memory True --resume /home/lkhagvadorj/Temuujin/TextBPN-Plus-Plus/model/Totaltext/TextBPN_resnet50_285.pth

