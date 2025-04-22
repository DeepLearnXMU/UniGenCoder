export PATH=$PATH:/home/user/UniGenCoder
echo $PATH
python sh/run_exp.py --model_tag codet5_base --task concode --sub_task none --res_dir results/codeT5 --model_dir saved_models/codeT5 --summary_dir tensorboard/codeT5 --gpu 5 --decoder_type seq --seed 4096
