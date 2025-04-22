DATE=$(date "+%Y%m%d%H%M%S")
# DATE=20230611000436
SEED=1234
trg=150
lr=10
rule_begin=true
OUTPUT_DIR=saved_models/codeT5_tune/concode/codet5_base_all_lr${lr}_bs16_src320_trg${trg}_pat3_e30
CACHE_DIR=${OUTPUT_DIR}/cache_data
OUTPUT_DIR=${OUTPUT_DIR}_${DATE}_${SEED}
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}
gpu=$1

cmd="CUDA_VISIBLE_DEVICES=${gpu} python run_tune.py \
--do_train --do_eval --do_eval_bleu --do_test \
--task concode --sub_task none --model_type codet5 --data_num -1 --num_train_epochs 60 --warmup_steps 10 \
--learning_rate ${lr}e-5 \
--patience 8 --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir data \
--cache_path ${CACHE_DIR} \
--output_dir ${OUTPUT_DIR} \
--summary_dir tensorboard/codeT5_tune --save_last_checkpoints --always_save_model \
--res_dir ${RES_DIR} \
--res_fn results/codeT5_tune/concode_codet5_base.txt \
--train_batch_size 16 --eval_batch_size 5 \
--max_source_length 320 --max_target_length ${trg} \
--gradient_accumulation_steps 1 --decoder_type tree \
--load_model_path  ${backbone_model_path} \
--seed ${SEED} \
--avg_checkpoint_path None \
--train_decode_label_path ${train_decode_label_path} \
--train_seq_output ${train_seq_output} \
--train_tree_output ${train_tree_output} \
--dev_decode_label_path ${dev_decode_label_path} \
--dev_seq_output ${dev_seq_output} \
--dev_tree_output ${dev_tree_output} \
--test_seq_output ${test_seq_output} \
--test_tree_output ${test_tree_output} \
--tune_on_label \
--train_filename data/concode/train.json \
--dev_filename data/concode/dev.json \
--complex_classifier \
--sample_file ${sample_file} \
--reweight \
--margin_loss \
--margin 2.0"

if [[ ${rule_begin} == true ]]; then
  cmd="${cmd} --rule_begin"
fi
cmd="${cmd} 2>&1 | tee ${LOG}"

echo ${cmd}
eval ${cmd}