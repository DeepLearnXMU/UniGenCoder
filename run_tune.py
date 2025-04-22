import time
import torch
import numpy as np
import os
import argparse
import multiprocessing
import torch
import pickle
import logging
from tqdm import tqdm, trange
import matplotlib
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu, _bleu_sentence
from tqdm.contrib import tzip
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import TreeSeqCodeT5
from utils import get_filenames, get_elapse_time, load_and_cache_gen_tune_data
from configs import add_args, set_seed, set_dist, model_args
from asdl.asdl import ASDLGrammar
from asdl.lang.java.java_transition_system import JavaTransitionSystem
from gpu_mem_track import MemTracker
from run_gen import KeepKBestModel
import math
import numpy as np

matplotlib.use('Agg')
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
        source_ids, seq_code_ids, tree_code_ids, decode_type_label, example_weight = batch
        target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = None, None, None, None, None
        tree_labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        tree_target_mask = None
        seq_target_mask = None

        # decode_type_label = target_ids[:, 0] == tokenizer.convert_tokens_to_ids("<seq>") # true代表seq，false代表tree
        first_token_loss_mask = torch.ones_like(decode_type_label).bool().to(args.device)
        outputs = model(input_ids=source_ids, attention_mask=source_mask, seq_labels=target_ids, tree_labels=tree_labels, 
                        seq_decoder_attention_mask=seq_target_mask, tree_decoder_attention_mask=tree_target_mask, 
                        decode_type_label=decode_type_label,first_token_loss_mask=first_token_loss_mask,
                        seq_code_ids=seq_code_ids, tree_code_ids=tree_code_ids,
                        example_weight=example_weight)
        loss = outputs[0]

        eval_loss += loss.item()
        batch_num += 1
        if args.no_ppl:
            break
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria, decode_type=None):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_decode_type = []
    bleu, codebleu = 0.0, 0.0
    if os.path.exists("saved_models/frontier_field2mask_multitask_bool_dict.bin"):
        with open("saved_models/frontier_field2mask_multitask_bool_dict.bin", "rb") as f:
            frontier_field2mask_dict = pickle.load(f)
            under_record = False
            logger.info("load from saved_models/frontier_field2mask_multitask_bool_dict.bin")
    else:
        frontier_field2mask_dict = {}
        under_record = True
        logger.info("create saved_models/frontier_field2mask_multitask_bool_dict.bin")
    # gpu_tracker = MemTracker()
    tree_decode_count = 0
    tree_probs = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        # gpu_tracker.track()
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, seq_code_ids, tree_code_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if not args.random_select:
                preds, probs = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        beam_size=args.beam_size,
                                        max_length=args.max_target_length,
                                        frontier_field2mask_dict=frontier_field2mask_dict,
                                        under_record=under_record,
                                        tokenizer=tokenizer,
                                        decode_type=decode_type,
                                        seq_code_ids=seq_code_ids,
                                        tree_code_ids=tree_code_ids,
                                        )
            else:
                preds, probs = torch.randint(0, 2, (source_ids.size(0), )).to(args.device), torch.rand((source_ids.size(0), 2)).to(args.device)
            decoder_type_labels = preds
            tree_probs.extend(probs[:, 1].cpu().numpy().tolist())
            pred_decode_type.extend(decoder_type_labels.cpu().numpy().tolist())
            tree_decode_count += decoder_type_labels.sum().item()
        if args.no_bleu:
            break

    logger.info("seq_decode_count: {}, tree_decode_count: {}".format(len(eval_data) - tree_decode_count, tree_decode_count))
    # save list: pred_decode_type to os.path.join(args.res_dir, "test_{}.select".format(criteria))
    np.savetxt(os.path.join(args.res_dir, "test_{}.select".format(criteria)), np.array(pred_decode_type), fmt="%d")

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    if decode_type is None:
        output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
        gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
        src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    else:
        output_fn = os.path.join(args.res_dir, "test_{}_{}.output".format(criteria, decode_type))
        gold_fn = os.path.join(args.res_dir, "test_{}_{}.gold".format(criteria, decode_type))
        src_fn = os.path.join(args.res_dir, "test_{}_{}.src".format(criteria, decode_type))

    pickle.dump(frontier_field2mask_dict, open("saved_models/frontier_field2mask_multitask_bool_dict.bin", "wb"))
    logger.info("save to saved_models/frontier_field2mask_multitask_bool_dict.bin")
    frontier_field2mask_dict = None
    torch.cuda.empty_cache()

    seq_fn = args.dev_seq_output if split_tag == 'dev' else args.test_seq_output
    tree_fn = args.dev_tree_output if split_tag == 'dev' else args.test_tree_output
    pred_seq_nls = [line.strip() for line in open(seq_fn).readlines()]
    pred_tree_nls = [line.strip() for line in open(tree_fn).readlines()]
    if args.debug:
        pred_tree_nls = pred_tree_nls[:100]
        pred_seq_nls = pred_seq_nls[:100]
    pred_nls = [pred_seq_nls[index] if pred_decode_type[index] == 0 else pred_tree_nls[index] for index in range(len(pred_seq_nls))]
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
    
    bleu_records_tree = _bleu_sentence(gold_fn, tree_fn)
    bleu_records_seq = _bleu_sentence(gold_fn, seq_fn)
    decode_type_acc = 0
    total_count = 0
    true_tree_probs = []
    for bleu_tree, bleu_seq, pred, tree_prob in zip(bleu_records_tree, bleu_records_seq, pred_decode_type, tree_probs):
        bleu_diff = bleu_seq[0] - bleu_tree[0]
        if bleu_diff<0 and pred==1:
            decode_type_acc += 1
            true_tree_probs.append(tree_prob)
        elif bleu_diff>0 and pred==0:
            decode_type_acc += 1
        if bleu_diff!=0:
            total_count += 1
    decode_type_acc = decode_type_acc / total_count
    logger.info("total_count: {}".format(total_count))
    bins = np.arange(0.0, 1.1, 0.1)
    counts, edges = np.histogram(tree_probs, bins)
    logger.info("tree_labels counts: {}, edges: {}".format(counts, edges))
    counts, edges = np.histogram(true_tree_probs, bins)
    logger.info("true_tree_labels counts: {}, edges: {}".format(counts, edges))

    if args.task == 'summarize':
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    else:
        bleu = round(_bleu(gold_fn, output_fn), 2)
        if args.task in ['concode', 'translate', 'refine']:
            codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu, 'decode_type_acc': decode_type_acc * 100}
    if args.task == 'concode':
        result['codebleu'] = codebleu * 100

    logger.info("***** Eval results " + (decode_type if decode_type is not None else '') + "*****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def multitask_opts(parser):
    parser.add_argument('--train_decode_label_path', type=str, default=None, help='path to decoder label')
    parser.add_argument('--dev_decode_label_path', type=str, default=None, help='path to decoder label')
    parser.add_argument('--train_seq_output', type=str, default=None, help='path to train seq output', required=True)
    parser.add_argument('--train_tree_output', type=str, default=None, help='path to train tree output', required=True)
    parser.add_argument('--dev_seq_output', type=str, default=None, help='path to dev seq output', required=True)
    parser.add_argument('--dev_tree_output', type=str, default=None, help='path to dev tree output', required=True)
    parser.add_argument('--test_seq_output', type=str, default=None, help='path to test seq output', required=True)
    parser.add_argument('--test_tree_output', type=str, default=None, help='path to test tree output', required=True)
    parser.add_argument('--over_sample', action='store_true', help='whether to over sample')
    parser.add_argument('--down_sample', action='store_true', help='whether to down sample')
    parser.add_argument('--reweight', action='store_true', help='whether to reweight')
    parser.add_argument('--complex_classifier', action='store_true', help='whether to use complex classifier')
    parser.add_argument('--margin_loss', action='store_true', help='whether to use margin loss')
    parser.add_argument('--margin', type=float, default=1.0, help='margin for margin loss')
    parser.add_argument('--classifier_input_argument',default=1, type=int, help='argument as input for classifier')
    parser.add_argument('--sample_file', type=str, default=None, help='path to sample file')
    parser.add_argument('--random_select', action='store_true', help='whether to random select')


def main():
    parser = argparse.ArgumentParser()
    multitask_opts(parser)
    model_args(parser)
    args = parser.parse_args()
    
    if args.task in ['summarize']:
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'

    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)

    grammar = ASDLGrammar.from_text(open('asdl/lang/java/java_asdl.txt').read(), 'program')
    transition_system = JavaTransitionSystem(grammar)
    vocab = None
    model = TreeSeqCodeT5(args, vocab, transition_system)
    tokenizer = model.tokenizer
    data_load_function = load_and_cache_gen_tune_data

    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(1)# if args.debug else args.cpu_cont) # args.cpu_cont
    train_filename, dev_filename, test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if args.test_filename is None:
        args.test_filename = test_filename
    if args.dev_filename is None:
        args.dev_filename = dev_filename
    if args.train_filename is None:
        args.train_filename = train_filename
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train and (args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path)):
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data, class_ratio = data_load_function(args, args.train_filename, pool, tokenizer, 'train', multitask=True, decode_label_path=args.train_decode_label_path)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        if args.debug:
            train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                        num_workers=4, pin_memory=True)
        
        if args.tune_on_label:
            # 冻结分类器之外的所有参数
            for name, param in model.named_parameters():
                if 'decode_classifier' not in name:
                    param.requires_grad = False

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], # and p.requires_grad
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
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
        global_step, best_bleu_em, best_acc, best_ppl = 0, -1, 0, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt, not_acc_inc_cnt = 0, 0, 0 if args.do_eval_bleu else 1e6
        kbModels_bleu = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-bleu'))
        kbModels_acc = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-acc'))
        klModels = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-last'))

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            count, limit_count, limit_count_use, use_sum = 0, 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                if args.no_train:
                    break
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, seq_code_ids, tree_code_ids, decode_type_label, example_weight = batch
                target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = None, None, None, None, None
                tree_labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                tree_target_mask = None
                seq_target_mask = None
                # 如果batch里所有样本标签一致，并且args.margin_loss为True，则跳过这个batch
                # if args.margin_loss:
                #     if len(set(decode_type_label.tolist())) == 1:
                #         continue

                # target_ids[:, 0] = tokenizer.convert_tokens_to_ids("<tree>")
                # decode_type_label = target_ids[:, 0] == tokenizer.convert_tokens_to_ids("<seq>") # true代表seq，false代表tree
                first_token_loss_mask = torch.ones_like(decode_type_label).bool().to(args.device)
                outputs = model(input_ids=source_ids, attention_mask=source_mask, seq_labels=target_ids, tree_labels=tree_labels, 
                                seq_decoder_attention_mask=seq_target_mask, tree_decoder_attention_mask=tree_target_mask, 
                                decode_type_label=decode_type_label, first_token_loss_mask=first_token_loss_mask, class_ratio=class_ratio,
                                seq_code_ids=seq_code_ids, tree_code_ids=tree_code_ids,
                                example_weight=example_weight)
                loss = outputs[0]
                if args.count_decode_label:
                    pred_decode_labels = outputs[3]
                    limit_pred_decode_labels = outputs[4]
                    count += (pred_decode_labels == target_ids[:, 0]).sum()
                    limit_count += (limit_pred_decode_labels == target_ids[:, 0]).sum()
                    limit_count_use += ((limit_pred_decode_labels == target_ids[:, 0])*first_token_loss_mask).sum()
                    use_sum += first_token_loss_mask.sum()
                    continue

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(loss.item(), 3)))
            if args.count_decode_label:
                logger.info("count: {}, train_data: {}".format(count, len(train_data)))
                logger.info("limit_count: {}, train_data: {}".format(limit_count, len(train_data)))
                logger.info("limit_count_use: {}, use_sum: {}".format(limit_count_use, use_sum))
                exit(0)

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev', multitask=True, decode_label_path=args.dev_decode_label_path)
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
                    # if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt, not_acc_inc_cnt]]):
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d, and not_acc_inc_cnt=%d \n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt, not_acc_inc_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=False) # , sample_number=50
                    dev_bleu_em = 0
                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em, dev_acc = result['bleu'], result['em'], result['decode_type_acc']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
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
                            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d, and not_acc_inc_cnt=%d \n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt, not_acc_inc_cnt)
                            logger.info(early_stop_str)
                            fa.write(early_stop_str)
                            break
                    kbModels_bleu.add(model, dev_bleu_em, cur_epoch)
                    if dev_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  [%d] Best acc: %.2f", cur_epoch, dev_acc)
                        logger.info("  " + "*" * 20)
                        best_acc = dev_acc
                        fa.write("[%d] Best acc changed into %.2f\n" % (cur_epoch, best_acc))
                        # Save best checkpoint for best acc
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best acc model into %s", output_model_file)
                    else:
                        not_acc_inc_cnt += 1
                        logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                        fa.write(
                            "[%d] Best acc (%.2f) does not drop changed for %d epochs, cur acc: %.2f \n" % (
                                cur_epoch, best_acc, not_acc_inc_cnt, dev_acc))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d, and not_acc_inc_cnt=%d \n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt, not_acc_inc_cnt)
                            logger.info(early_stop_str)
                            fa.write(early_stop_str)
                            break
                    kbModels_acc.add(model, dev_acc, cur_epoch)
            
            if args.tune_on_label:
                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        logger.info("  %s", name)
                        logger.info("  %s", str(param.data))
            
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        criteria_list = ['best-bleu', 'best-acc']
        if args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path):
            criteria_list.append('bleu-average.bin')
            criteria_list.append('acc-average.bin')
        logger.info("  criteria_list = %s", str(criteria_list))
        for criteria in criteria_list:
            if criteria == 'bleu-average.bin': # test after train mode
                file = os.path.join(args.output_dir, 'checkpoint-best-bleu/average.bin')
                if not os.path.exists(file):
                    status = os.system('python average_checkpoints.py --path '+os.path.join(args.output_dir, 'checkpoint-best-bleu')+' --output '+os.path.join(args.output_dir, 'checkpoint-best-bleu'))
                    if status == 0:
                        logger.info("success generate %s", file)
                    else: # if happen in pycharm run/debug, don't care, it will work well in the server
                        logger.error("fail to generate %s", file)
            elif criteria == 'acc-average.bin':
                file = os.path.join(args.output_dir, 'checkpoint-best-acc/average.bin')
                if not os.path.exists(file):
                    status = os.system('python average_checkpoints.py --path '+os.path.join(args.output_dir, 'checkpoint-best-acc')+' --output '+os.path.join(args.output_dir, 'checkpoint-best-acc'))
                    if status == 0:
                        logger.info("success generate %s", file)
                    else: # if happen in pycharm run/debug, don't care, it will work well in the server
                        logger.error("fail to generate %s", file)
            else:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if args.avg_checkpoint_path is not None and os.path.exists(args.avg_checkpoint_path):
                file = args.avg_checkpoint_path # test-only mode 
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = data_load_function(args, args.test_filename, pool, tokenizer, args.test_split_tag,
                                                               only_src=True, is_sample=False)
            
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em, test_acc = result['bleu'], result['em'], result['decode_type_acc']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f, acc: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu, test_acc)
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


if __name__ == '__main__':
    main()
