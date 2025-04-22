import argparse
import logging
import torch
import os
import time
import multiprocessing
import numpy as np
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib import tzip
from configs import model_args, set_seed, set_dist
from utils import load_and_cache_gen_tree_data, load_and_cache_gen_data, get_elapse_time, get_filenames
from models import TreeCodeT5, build_or_load_gen_model
from asdl.asdl import ASDLGrammar
from asdl.lang.java.java_transition_system import JavaTransitionSystem
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from run_gen import eval_bleu_epoch, eval_ppl_epoch, KeepKBestModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def seq_distll_loss(student_logits, teacher_logits, target_mask):
    # student_logits: batch_size, seq_len, vocab_size
    # teacher_logits: batch_size, seq_len, vocab_size
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=-1),
        torch.nn.functional.softmax(teacher_logits, dim=-1),
        reduction='none'
    )
    loss = loss.sum(-1) * target_mask
    # loss = loss.mean()
    loss = loss.sum() / loss.size(0)
    return loss

def vanilla_distill_loss(args, student_logits, teacher_logits, teacher_token_mask, student_target_mask, match_index, teacher_token, student_token, m_fa,tokenizer=None):
    # Take out the teacher_logits at the corresponding teacher_token_mask position as target_teacher_logit
    # teacher:1106 276 n*2 288({) 289(})，第一个单词bpe结果不同，比如teacher输入class c function，function前有空格
    # student:1 2
    # student数据中的>>已经全部改成了> >
    # teacher_token_mask[:, 2:4] = torch.zeros(teacher_logits.shape[0], 2, dtype=bool)
    # 去</s>
    error = False
    teacher_valid_mask = teacher_token_mask & (teacher_token != 2)
    student_valid_mask = student_target_mask & (student_token != 2)
    # 去头 and teacher去尾：
    if m_fa is None:
        for index, (teacher_head, teacher_tail, student_head, student_tail) in enumerate(match_index):
            teacher_valid_mask[index, :teacher_head] = False
            teacher_valid_mask[index, teacher_tail:] = False
            student_valid_mask[index, :student_head] = False
            student_valid_mask[index, student_tail:] = False
    else:
        teacher_tail_index = teacher_token_mask.sum(-1)
        student_tail_index = student_target_mask.sum(-1)
        for index, head_pair in enumerate(match_index):
            teacher_head, student_head = head_pair
            teacher_head = torch.nonzero(torch.cumsum(teacher_valid_mask[index], dim=-1) == teacher_head+1)[0]
            teacher_tail = torch.nonzero(torch.cumsum(teacher_token_mask[index], dim=-1) == teacher_tail_index[index])[0]
            teacher_cumsum = torch.cumsum(teacher_valid_mask[index], dim=-1)
            teacher_valid_mask[index, :teacher_head] = False
            student_valid_mask[index, :student_head] = False
            length_diff = len(teacher_token[index][teacher_valid_mask[index]]) - len(student_token[index][student_valid_mask[index]])
            if length_diff == 1:
                if teacher_token[index, teacher_tail] == 2:
                    teacher_valid_mask[index, teacher_tail - 1:] = False
                    teacher_tail_record = teacher_tail - 1
                else:
                    teacher_valid_mask[index, teacher_tail:] = False
                    teacher_tail_record = teacher_tail
                student_tail = teacher_cumsum[teacher_tail] - teacher_cumsum[teacher_head] + student_head
            elif length_diff == 2:
                teacher_valid_mask[index, teacher_tail-1:] = False
                teacher_tail_record = teacher_tail - 1
                student_tail = teacher_cumsum[teacher_tail] - teacher_cumsum[teacher_head] + student_head
            elif length_diff <= 0:
                student_tail = teacher_cumsum[teacher_tail] - teacher_cumsum[teacher_head] + student_head + 1
                teacher_tail_record = teacher_tail + 1
            else:
                print(length_diff)
                raise ValueError("length_diff out of range")
            student_valid_mask[index, student_tail:] = False
            if not teacher_token[index][teacher_valid_mask[index]].equal(student_token[index][student_valid_mask[index]]):
                print(length_diff)
                print(teacher_tail)
                print(student_tail)
                print(teacher_token[index])
                print(teacher_token[index][teacher_valid_mask[index]])
                print(student_token[index])
                print(student_token[index][student_valid_mask[index]])
                error = True

            record_list = [teacher_head.data.item(), teacher_tail_record.data.item(), student_head.data.item(), student_tail.data.item()]
            record_list = [str(e) for e in record_list]
            m_fa.write(' '.join(record_list) + '\n')

        for index in range(student_token.shape[0]):
            if not teacher_token[index][teacher_valid_mask[index]].equal(
                    student_token[index][student_valid_mask[index]]):
                error = True

    target_teacher_logits = teacher_logits[teacher_valid_mask]
    target_student_logits = student_logits[student_valid_mask] # valid_token_num, vocab_size

    # for index in range(student_token.shape[0]):
    #     if not teacher_token[index][teacher_valid_mask[index]].equal(student_token[index][student_valid_mask[index]]):
    #         raise ValueError("Tree node and sequence token mismatch")
    #     token_id = teacher_token[index][teacher_valid_mask[index]]
    #     sentence_token = []
    #     for j in range(teacher_token[index][teacher_valid_mask[index]].shape[0]):
    #         sentence_token.append(tokenizer.decode(token_id[j]))
    #     print(sentence_token)

    if not error:
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(target_student_logits, dim=-1),
            torch.nn.functional.softmax(target_teacher_logits, dim=-1),
            reduction='none'
        ) # valid_token_num
        # sum and average the loss
        # 判断tensor里每一项是否小于阈值
        # 教师模型ground truth位置预测正确率达到一定程度才将学生模型拉近教师模型。阈值=0.4
        # target_index_id = k, target_student_logits= k*32100
        # print(loss.sum(-1))
        # print(torch.gather(torch.nn.functional.softmax(target_student_logits, dim=-1), 1, teacher_index_id.unsqueeze(-1)))
        # print(torch.gather(torch.nn.functional.softmax(target_teacher_logits, dim=-1), 1, teacher_index_id.unsqueeze(-1)))
        loss = loss.sum(-1)
        teacher_index_id = teacher_token[teacher_valid_mask]
        mask = torch.gather(torch.nn.functional.softmax(target_teacher_logits, dim=-1), 1, teacher_index_id.unsqueeze(-1)) > args.distill_threshold
        loss = loss * mask.squeeze(-1)
        loss = loss.sum() / loss.size(0)
    else:
        loss = 0

    return loss


# todo: check the parallelism of the following code
def match_distill_loss(args, student_logits,teacher_rule_logits, teacher_token_logits,
                       teacher_token_mask, student_target_mask, match_index,
                       teacher_token, student_token, teacher_rule_row, m_fa=None,tokenizer=None):
    teacher_valid_mask = teacher_token_mask & (teacher_token != 2)
    student_valid_mask = student_target_mask & (student_token != 2)
    # 去头 and teacher去尾：
    if m_fa is None:
        for index, (teacher_head, teacher_tail, student_head, student_tail) in enumerate(match_index):
            teacher_valid_mask[index, :teacher_head] = False
            teacher_valid_mask[index, teacher_tail:] = False
            student_valid_mask[index, :student_head] = False
            student_valid_mask[index, student_tail:] = False

    target_teacher_logits = teacher_token_logits[teacher_valid_mask]
    target_student_logits = student_logits[student_valid_mask] 

    rule_prob_dist = torch.nn.functional.softmax(teacher_rule_logits, dim=-1)
    target_rule_prob = torch.gather(rule_prob_dist, dim=2, index=teacher_rule_row.unsqueeze(2)).squeeze(2)
    target_rule_prob = target_rule_prob.masked_fill(teacher_token_mask, 1)
    acum_prob = torch.cumprod(target_rule_prob, dim=1)
    print(acum_prob)
    # get acum_head from acum_prob with teacher_token_mask
    acum_head = torch.cat((torch.ones(teacher_token_mask.size()[0], 1), acum_prob[teacher_token_mask][:, -1]), dim=1)
    print(acum_head)
    # get acum_tail from acum_prob with 
    acum_tail = torch.cat((torch.ones(teacher_token_mask.size()[0], 1), acum_prob[:, -1]), dim=1)[teacher_token_mask]
    print(acum_tail)
    sub_path_prob = acum_tail / acum_head
    print(sub_path_prob)

    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(target_student_logits, dim=-1),
        torch.nn.functional.softmax(target_teacher_logits, dim=-1) * sub_path_prob, # todo: check element-wise multiply
        reduction='none'
    )
    
    # sum and average the loss
    # loss = loss.sum(-1)
    # teacher_index_id = teacher_token[teacher_valid_mask]
    # mask = torch.gather(torch.nn.functional.softmax(target_teacher_logits, dim=-1), 1, teacher_index_id.unsqueeze(-1)) > args.distill_threshold
    # loss = loss * mask.squeeze(-1)
    loss = loss.sum() / loss.size(0)
    return loss


def distill_opts(parser):
    parser.add_argument("--teacher_model_path", type=str)
    parser.add_argument("--distill_type", type=str, choices=['vanilla', 'match', 'seq'])
    parser.add_argument('--distill_threshold', type=float, default=0.5)


def main():
    # load arguments
    parser = argparse.ArgumentParser()
    distill_opts(parser)
    model_args(parser)
    args = parser.parse_args()
    args.lang = 'java'
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    # load teacher model
    grammar = ASDLGrammar.from_text(open('asdl/lang/java/java_asdl.txt').read(), 'program')
    transition_system = JavaTransitionSystem(grammar)
    if args.distill_type == 'seq': # load a seq model as teacher
        config, teacher_model, tokenizer = build_or_load_gen_model(args)
    else:
        teacher_model = TreeCodeT5(args, vocab=None, transition_system=transition_system)
    if args.teacher_model_path is not None:
        logger.info("Reload teacher model from {}".format(args.teacher_model_path))
        teacher_model.load_state_dict(torch.load(args.teacher_model_path))
    teacher_model.to(args.device)
    
    # init student model
    config, student_model, tokenizer = build_or_load_gen_model(args)
    if args.load_model_path is not None:
        logger.info("Reload student model from {}".format(args.load_model_path))
        student_model.load_state_dict(torch.load(args.load_model_path))
    student_model.to(args.device)

    # pool = multiprocessing.Pool(1)
    pool = multiprocessing.Pool(1 if args.debug else args.cpu_cont) # args.cpu_cont
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    m_fa = None
    # m_fa = open('bin/train_debug_match_index_new.txt', 'w') if args.debug else open('bin/train_match_index_new.txt', 'w')

    if args.do_train and (args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path)):
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)
            
        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_tree_data(args, args.train_filename, pool, tokenizer, 'train', distill=True)
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
            {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
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
            generation_tr_loss, distill_tr_loss = 0, 0
            teacher_tr_loss = 0
            student_model.train()
            teacher_model.eval()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row, match_index = batch
                labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                teacher_target_mask = (app_rule_mask_row + gen_token_mask_row).bool()
                student_target_mask = target_ids.ne(tokenizer.pad_token_id)
                with torch.no_grad():
                    if args.distill_type == 'seq':
                        teacher_outputs = teacher_model(input_ids=source_ids, attention_mask=source_mask,
                                            labels=target_ids, decoder_attention_mask=student_target_mask)
                    else:
                        teacher_outputs = teacher_model(input_ids=source_ids, attention_mask=source_mask,
                                            labels=labels, decoder_attention_mask=teacher_target_mask)
                        teacher_token_logits = teacher_outputs[1]
                        teacher_rule_logits = teacher_outputs[2]
                    # teacher_outputs[1] = teacher_outputs[1].detach()
                    # teacher_outputs[2] = teacher_outputs[2].detach()
                student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=student_target_mask)
                # output = (gen_from_vocab_logits, apply_rule_logits, ) + decoder_outputs[1:] + (encoder_outputs, )
                # ((loss,) + output)
                generation_loss = student_outputs.loss
                if args.distill_type == 'vanilla':
                    distill_loss = vanilla_distill_loss(args, student_outputs.logits, teacher_token_logits, 
                                         gen_token_mask_row.bool(), student_target_mask, match_index, 
                                         token_row, target_ids, m_fa, tokenizer)
                elif args.distill_type == 'match':
                    distill_loss = match_distill_loss(args, student_outputs.logits, teacher_token_logits, teacher_rule_logits,
                                        gen_token_mask_row.bool(), student_target_mask, match_index, 
                                        token_row, target_ids, app_rule_idx_row)
                elif args.distill_type == 'seq':
                    distill_loss = seq_distll_loss(student_outputs.logits, teacher_outputs.logits.detach(), student_target_mask)
                lambda_ = 0.5 #args.lambda_
                # lambda_ = 0.9 * (1 - cur_epoch / args.num_train_epochs)
                # if lambda_ < 0:
                #     lambda_ = 0

                # lambda_ = 0

                loss = (1-lambda_) * generation_loss + lambda_ * distill_loss
                # loss = generation_loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    generation_loss = generation_loss / args.gradient_accumulation_steps
                    distill_loss = distill_loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                generation_tr_loss += generation_loss.item()
                distill_tr_loss += distill_loss.item()


                teacher_tr_loss += teacher_outputs.loss.item()


                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)
                    generation_train_loss = generation_tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)
                    distill_train_loss = distill_tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)


                    teacher_train_loss = teacher_tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)


                    # 打印每一个batch单独的loss
                    bar.set_description("[{}] Generation loss {}, Distill loss {}, Teacher loss {}".format(cur_epoch, round(generation_train_loss, 6), round(distill_train_loss, 6), round(teacher_train_loss, 6)))
                    # generation_tr_loss, distill_tr_loss = 0, 0
                    # teacher_tr_loss = 0

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, student_model, tokenizer)
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
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)
                    klModels.add(student_model, cur_epoch, cur_epoch)

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
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
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
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                    only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, 'dev', 'e%d' % cur_epoch)
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
                            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
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
                    kbModels.add(student_model, dev_bleu_em, cur_epoch)
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))
        if m_fa is not None:
            m_fa.close()
            logger.info("finish write m_fa")

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        criteria_list = ['best-bleu'] if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-bleu/pytorch_model.bin')) else ['best-ppl']
        criteria_list.append('average.bin')
        for criteria in criteria_list:
            if criteria == 'average.bin':
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
                file = args.avg_checkpoint_path
            logger.info("Reload model from {}".format(file))
            student_model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, 'test', criteria)
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
    
def evaluation():
    criteria = "best-bleu"
    res_dir = 'saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_debug_tree/prediction'
    output_fn = os.path.join(res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(res_dir, "test_{}.gold".format(criteria))
    dev_accs, predictions = [], []
    with open(output_fn, 'r') as f, open(gold_fn, 'r') as f1:
        pred_nls = f.readlines()
        golds = f1.readlines()
        for pred_nl, gold in tzip(pred_nls, golds):
            dev_accs.append(pred_nl.strip() == gold.strip())
            if pred_nl.strip() == gold.strip():
                print(gold)
    
    bleu = round(_bleu(gold_fn, output_fn), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, 'java')
    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu, 'codebleu': codebleu*100}
    print(result)


if __name__ == "__main__":
    main()