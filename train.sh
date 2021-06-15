for i in {0..11}
do
  bsub -n 1 -W 26:00 -R "rusage[mem=15000,scratch=15000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=24000]" -J hr python3 train.py --lr_index=$i --wd_index=$i
done
