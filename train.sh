time_array=([1]=24:00 [2]=24:00 [3]=24:00 [4]=24:00)
eps_array=([1]=0.0 [2]=0.5 [3]=1.0)

for i in {1..3}
do
    for j in {2..4}
    do
	bsub -n 1 -W ${time_array[$j]} -R "rusage[mem=15000,scratch=15000,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=10000]" -J hr python3 train.py --epsilon=${eps_array[$i]} --train_steps=30000 --num_scales=$j
    done
done
