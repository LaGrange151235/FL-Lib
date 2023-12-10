home='/home/ubuntu/FL-Lib'

remarks="FedAvg"
#remarks="FedProx"
#remarks="FedNova"
#remarks="MOON"
#remarks="SCAFFOLD"

n_clients=4
#test_mod="inline"
test_mod="independent"
C_mod="random"
#C_mod="fast"
C=1
iid_mod=3
sync_frequency=200

model="CNN_Cifar10"
#model="CNN_EMnist"
#model="ResNet20_Cifar10"
#model="ResNet18_Cifar10"
#model="LSTM_KWS"

dataset="Cifar10"
#dataset="KWS"
#dataset="EMnist"

cuda="0,1,2,3"
n_cuda=4

trial_no=$(ls $home/Logs/ | wc -l)
log_dir=$home/Logs/${trial_no}_$remarks
mkdir -p $log_dir/code_snapshot 
cp $home/*.sh $log_dir/code_snapshot
cp $home/*.py $log_dir/code_snapshot
cp -r $home/models $log_dir/code_snapshot
rm -f Latest_Log && ln -s $log_dir Latest_Log
echo 'logs in '$log_dir

if [ "$remarks" = "MOON" ]
then 
        model="$model""_MOON"
fi

if [ "$test_mod" = "independent" ] 
then
        command="export CUDA_VISIBLE_DEVICES=$cuda && /opt/conda/bin/python3.9 $home/$remarks/server_test_process.py \
                --port=$((10000+trial_no)) \
                --trial_no=$trial_no \
                --remarks=$remarks \
                --model_name=$model \
                --dataset_name=$dataset \
                --home=$home \
                "
        echo $command
        nohup ssh localhost $command >> $log_dir/test.log 2>&1 &
        sleep 10
fi

command="export CUDA_VISIBLE_DEVICES=$cuda && /opt/conda/bin/python3.9 $home/$remarks/server_process.py \
        --home=$home \
        --port=$((20000+trial_no)) \
        --client_number=$((n_clients)) \
        --trial_no=$trial_no \
        --remarks=$remarks \
        --test_mod=$test_mod \
        --test_process_port=$((10000+trial_no)) \
        --model_name=$model \
        --dataset_name=$dataset \
        --C=$C \
        --C_mod=$C_mod \
        "
echo "launch server:"
echo $command
echo 
nohup ssh localhost $command >> $log_dir/server.log 2>&1 &
sleep 10

cuda_ids=0

client_serial_num=0
for ((j=0; j<$n_clients; j++))
do  
	command="export CUDA_VISIBLE_DEVICES=$((cuda_ids%n_cuda)) && /opt/conda/bin/python3.9 $home/$remarks/worker_process.py \
                --home=$home \
	    	--server_ip=localhost \
                --server_port=$((20000 + trial_no)) \
                --world_size=$n_clients \
                --rank=$client_serial_num \
                --trial_no=$trial_no \
	        --remarks=$remarks \
                --model_name=$model \
                --dataset_name=$dataset \
                --is_iid=$iid_mod \
                --sync_frequency=$sync_frequency
                "
	echo "launch client "${client_serial_num}": "
	echo $command
	nohup ssh localhost $command >> $log_dir/client_${client_serial_num}.log 2>&1 &
	client_serial_num=$((client_serial_num+1))
        cuda_ids=$((cuda_ids+1))
        sleep 0.5
done
echo