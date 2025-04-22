DATE=$(date "+%Y%m%d%H%M%S")
SEED=1234
trg=150
lr=10
rule_begin=true
OUTPUT_DIR=saved_models/codeT5_tree/concode/codet5_base_all_lr${lr}_bs16_src320_trg${trg}_pat3_e30
CACHE_DIR=${OUTPUT_DIR}/cache_data
OUTPUT_DIR=${OUTPUT_DIR}_${DATE}_${SEED}
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

cmd="CUDA_VISIBLE_DEVICES=4 python run_gen.py \
--do_train --do_eval --do_eval_bleu --do_test \
--task concode --sub_task none --model_type codet5 --data_num -1 --num_train_epochs 30 --warmup_steps 1000 \
--learning_rate ${lr}e-5 \
--patience 3 --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir data \
--cache_path ${CACHE_DIR} \
--output_dir ${OUTPUT_DIR} \
--summary_dir tensorboard/codeT5_tree --save_last_checkpoints --always_save_model \
--res_dir ${RES_DIR} \
--res_fn results/codeT5_tree/concode_codet5_base.txt \
--train_batch_size 16 --eval_batch_size 16 \
--max_source_length 320 --max_target_length ${trg} \
--gradient_accumulation_steps 2 --decoder_type tree \
--load_model_path None \
--seed ${SEED} \
--avg_checkpoint_path None"


if [[ ${rule_begin} == true ]]; then
  cmd="${cmd} --rule_begin"
fi
cmd="${cmd} 2>&1 | tee ${LOG}"

echo ${cmd}
eval ${cmd}