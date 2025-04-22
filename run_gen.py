# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import multiprocessing
import time
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model, TreeCodeT5
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, load_and_cache_gen_tree_data, load_and_cache_gen_bi_tree_data
from configs import add_args, set_seed, set_dist
from asdl.asdl import ASDLGrammar
from asdl.lang.java.java_transition_system import JavaTransitionSystem
from gpu_mem_track import MemTracker

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        if args.decoder_type == "tree":
            if args.task=='translate' and args.source_ast:
                src_rule, src_rule_mask, src_token, src_token_mask, \
                    target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row  = batch
                source_ids = (src_rule, src_rule_mask, src_token, src_token_mask)
                source_mask = (src_rule_mask + src_token_mask).bool()
            else:
                source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)

            labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
            target_mask = (app_rule_mask_row + gen_token_mask_row).bool()
        else:
            # source_ids, target_ids = batch
            source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = batch
            labels = None
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids if labels==None else labels, decoder_attention_mask=target_mask)
                loss = outputs[0] if args.decoder_type == "tree" else outputs.loss

        eval_loss += loss.item()
        batch_num += 1
        
        if args.no_ppl:
            break
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    if args.task == 'concode':
        frontier_field2mask_dict_path = "saved_models/frontier_field2mask_bool_dict.bin"
    elif args.task == 'translate' and args.sub_task == 'cs-java':
        frontier_field2mask_dict_path = "saved_models/frontier_field2mask_bool_dict_translate_java.bin"
    elif args.task == 'translate' and args.sub_task == 'java-cs':
        frontier_field2mask_dict_path = "saved_models/frontier_field2mask_bool_dict_translate_cs.bin"
        
    # 不存在或者第0轮，创建一个空的dict
    if os.path.exists(frontier_field2mask_dict_path) and criteria != 'e0':
        with open(frontier_field2mask_dict_path, "rb") as f:
            frontier_field2mask_dict = pickle.load(f)
            under_record = False
            logger.info("load from saved_models/{}".format(frontier_field2mask_dict_path))
    else:
        frontier_field2mask_dict = {}
        under_record = True
        logger.info("create saved_models/{}".format(frontier_field2mask_dict_path))
    # gpu_tracker = MemTracker()
    t = 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        if t > 35:
            break
        t += 1
        # gpu_tracker.track()
        if args.task=='translate' and args.source_ast:
            batch = tuple(t.to(args.device) for t in batch)
            src_rule, src_rule_mask, src_token, src_token_mask = batch
            source_ids = (src_rule, src_rule_mask, src_token, src_token_mask)
            source_mask = (src_rule_mask + src_token_mask).bool()
        else:
            # source_ids = batch[0].to(args.device)
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = batch
            tree_labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if args.decoder_type == "tree":
                    # try:
                    preds = []
                    preds = model.generate(source_ids, # 一条数据
                                           attention_mask=source_mask,
                                           beam_size=args.beam_size,
                                           max_length=args.max_target_length,
                                           frontier_field2mask_dict=frontier_field2mask_dict,
                                           under_record=under_record,
                                           tokenizer=tokenizer,
                                           args=args,
                                           tree_labels=tree_labels)
                    # break
                    # except Exception as e:
                    #     print("ERROR: ")
                    #     print(e)
                    top_preds = preds # 一个batch
                else:
                    preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                    top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds) # 一个数据集
        
        if args.no_bleu:
            break

    if args.task == 'translate' and args.decoder_type == 'tree':
        pred_nls = pred_ids
        # for l in pred_nls:
        #     print(l)
    else:
        pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
        
    if args.decoder_type == "tree":
        if args.task == 'concode' or args.task == 'translate':
            # if has 'class c { ' in pred_nl, detele it
            pred_nls = [pred_nl.replace('class c { ', '') for pred_nl in pred_nls]
            # if the number of '{' is not equal to the number of '}', delete the last '}'
            pred_nls = [pred_nl[:-1] if pred_nl.count('{') != pred_nl.count('}') and pred_nl[-1]=='}' else pred_nl for pred_nl in pred_nls]
        # elif args.task == 'translate' and args.sub_task == 'cs-java':
        #     pred_nls = re_organize_code('java', grammar, pred_nls)
        # elif args.task == 'translate' and args.sub_task == 'java-cs':
        #     pred_nls = re_organize_code('cs', grammar, pred_nls)

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    pickle.dump(frontier_field2mask_dict, open(frontier_field2mask_dict_path, "wb"))
    logger.info("save to saved_models/{}".format(frontier_field2mask_dict_path))
    frontier_field2mask_dict = None
    torch.cuda.empty_cache()

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in tzip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


class KeepKBestModel():
    def __init__(self, K, keep_path, worst_score=100):
        self.K = K
        self.models = []
        self.worst_score = worst_score
        self.keep_path = keep_path

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.models)

    def add(self, model, bleu_score, epoch_index):
        """
        Add a new hypothesis to the list.
        """
        if len(self) < self.K or bleu_score > self.worst_score:
            self.models.append((bleu_score, epoch_index))
            # 保存模型
            torch.save(model.state_dict(), os.path.join(self.keep_path, str(epoch_index) + '_' + str(bleu_score) + '.bin'))
            logger.info('save model to %s' % os.path.join(self.keep_path, str(epoch_index) + '_' + str(bleu_score) + '.bin'))
            if len(self) > self.K: # 如果超过了beam size, 则删除最差的那个
                sorted_next_scores = sorted([(s, idx, epoch) for idx, (s, epoch,) in enumerate(self.models)])
                del self.models[sorted_next_scores[0][1]]
                # 删除模型
                os.remove(os.path.join(self.keep_path, str(sorted_next_scores[0][2]) + '_' + str(sorted_next_scores[0][0]) + '.bin'))
                logger.info('remove model %s' % os.path.join(self.keep_path, str(sorted_next_scores[0][2]) + '_' + str(sorted_next_scores[0][0]) + '.bin'))
                self.worst_score = sorted_next_scores[1][0]
            else: # 如果没有超过beam size, 加完更新最低分后就不管了
                self.worst_score = min(bleu_score, self.worst_score)


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    data_load_function = load_and_cache_gen_data
    if args.decoder_type == 'seq':
        config, model, tokenizer = build_or_load_gen_model(args)
    elif args.decoder_type == 'tree':
        grammar_txt = 'asdl/lang/java/java_asdl.txt'
        if args.task == 'translate' and args.sub_task == 'cs-java':
            grammar_txt = 'asdl/lang/java/java_asdl_translate.txt'
            print('grammar_txt: ', grammar_txt)
            if args.source_ast:
                src_grammar_txt = 'asdl/lang/java/cs_asdl_translate.txt'
                src_grammar = ASDLGrammar.from_text(open(src_grammar_txt).read(), 'program')
                src_transition_system = JavaTransitionSystem(src_grammar)
            else:
                src_transition_system = None
        grammar = ASDLGrammar.from_text(open(grammar_txt).read(), 'program')
        transition_system = JavaTransitionSystem(grammar)
        vocab = None
        model = TreeCodeT5(args, vocab, transition_system, src_transition_system)
        tokenizer = model.tokenizer
        data_load_function = load_and_cache_gen_bi_tree_data if args.task=='translate' and args.source_ast else load_and_cache_gen_tree_data
    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(1 if args.debug else args.cpu_cont) # args.cpu_cont
    args.train_filename, args.dev_filename, test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if args.test_filename is None:
        args.test_filename = test_filename
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train and (args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path)):
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = data_load_function(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        if args.debug:
            train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                        num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        kbModels = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-bleu'))
        klModels = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-last'))

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                if args.no_train:
                    break
                batch = tuple(t.to(args.device) for t in batch)
                if args.decoder_type == "tree":
                    if args.task=='translate' and args.source_ast:
                        src_rule, src_rule_mask, src_token, src_token_mask, \
                            target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row  = batch
                        source_ids = (src_rule, src_rule_mask, src_token, src_token_mask)
                        source_mask = (src_rule_mask + src_token_mask).bool()
                    else:
                        source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = batch
                        source_mask = source_ids.ne(tokenizer.pad_token_id)
                    labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
                    target_mask = (app_rule_mask_row + gen_token_mask_row).bool()
                else:
                    # source_ids, target_ids= batch
                    source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = batch
                    labels = None
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids if labels==None else labels, decoder_attention_mask=target_mask)
                    loss = outputs[0] if args.decoder_type == "tree" else outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += target_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)
                    klModels.add(model, cur_epoch, cur_epoch)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    # eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev',
                    #                                                    only_src=True, is_sample=True) # , sample_number=50

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
                    kbModels.add(model, dev_bleu_em, cur_epoch)
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        criteria_list = ['best-bleu'] if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-bleu/pytorch_model.bin')) else ['best-ppl']
        if not args.no_average:
            criteria_list.append('average.bin')
            
        eval_examples, eval_data = data_load_function(args, args.test_filename, pool, tokenizer, args.test_split_tag,
                                                      only_src=True, is_sample=False)
        for criteria in criteria_list:
            if criteria == 'average.bin': # test after train mode
                file = os.path.join(args.output_dir, 'checkpoint-best-bleu/average.bin')
                if not os.path.exists(file):
                    status = os.system('python average_checkpoints.py --path '+os.path.join(args.output_dir, 'checkpoint-best-bleu')+' --output '+os.path.join(args.output_dir, 'checkpoint-best-bleu'))
                    if status == 0:
                        logger.info("success generate average.bin")
                    else:
                        logger.error("fail to generate average.bin")
            else:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if args.avg_checkpoint_path is not None and os.path.exists(args.avg_checkpoint_path):
                file = args.avg_checkpoint_path # test-only mode 
            logger.info("Reload model from {}".format(file))
            state_dict = torch.load(file)
            if not args.source_ast:
                unmatched_key = 'src_production_embed.weight'
                if unmatched_key in state_dict:
                    del state_dict[unmatched_key]
            model.load_state_dict(state_dict, strict=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, args.test_split_tag, criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
            if args.avg_checkpoint_path is not None and os.path.exists(args.avg_checkpoint_path):
                break # test-only mode 
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
