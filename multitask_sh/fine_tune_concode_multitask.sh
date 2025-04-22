export PATH=$PATH:/home/user/UniGenCoder
echo $PATH
python multitask_sh/run_exp.py --model_tag codet5_base --task concode --sub_task none --res_dir results/codeT5_multitask --model_dir saved_models/codeT5_multitask --summary_dir tensorboard/codeT5_multitask --gpu 6 --decoder_type tree
