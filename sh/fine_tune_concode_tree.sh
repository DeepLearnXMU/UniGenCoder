export PATH=$PATH:/home/user/UniGenCoder
echo $PATH
python sh/run_exp.py --model_tag codet5_base --task concode --sub_task none --res_dir results/codeT5_tree --model_dir saved_models/codeT5_tree --summary_dir tensorboard/codeT5_tree --gpu 3  --decoder_type tree
