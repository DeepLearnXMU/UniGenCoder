# DATE=$(date "+%Y%m%d%H%M%S")
DATE=${training_date}
SEED=1234
trg=150
lr=10
beam_size=1
test_dataset=train
rule_begin=false
OUTPUT_DIR=saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr${lr}_bs16_src320_trg${trg}_pat3_e30
CACHE_DIR=${OUTPUT_DIR}/cache_data
OUTPUT_DIR=${OUTPUT_DIR}_${DATE}
RES_DIR=${OUTPUT_DIR}/prediction_beamsize${beam_size}_${test_dataset}
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

cmd="CUDA_VISIBLE_DEVICES=$1 python run_multitask_distill.py \
--do_test \
--task concode --sub_task none --model_type codet5 --data_num -1 --num_train_epochs 30 --warmup_steps 1000 \
--learning_rate ${lr}e-5 \
--patience 3 --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir data \
--cache_path ${CACHE_DIR} \
--output_dir ${OUTPUT_DIR} \
--summary_dir tensorboard/codeT5_multitask_distill --save_last_checkpoints --always_save_model \
--res_dir ${RES_DIR} \
--res_fn results/codeT5_multitask_distill/concode_codet5_base.txt \
--train_batch_size 16 --eval_batch_size 200 \
--max_source_length 320 --max_target_length ${trg} \
--gradient_accumulation_steps 2 --decoder_type seq \
--load_model_path None \
--seed ${SEED} \
--avg_checkpoint_path None \
--no_average \
--beam_size ${beam_size} \
--test_split_tag ${test_dataset} \
--avg_checkpoint_path ${Your_model_path}"


if [[ ${rule_begin} == true ]]; then
  cmd="${cmd} --rule_begin"
fi
if [[ ${test_dataset} == train ]]; then
  cmd="${cmd} --test_filename data/concode/train.json"
elif [[ ${test_dataset} == dev ]]; then
  cmd="${cmd} --test_filename data/concode/dev.json"
else
  cmd=${cmd}
fi
cmd="${cmd} 2>&1 | tee ${LOG}"

echo ${cmd}
eval ${cmd}
