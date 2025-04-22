import time
import torch
import os
import argparse
import multiprocessing
import torch
import logging
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
from run_multitask import eval_bleu_epoch, eval_ppl_epoch
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model, TreeCodeT5, TreeSeqCodeT5
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, load_and_cache_gen_tree_data
from configs import add_args, set_seed, set_dist, model_args
from asdl.asdl import ASDLGrammar
from asdl.lang.java.java_transition_system import JavaTransitionSystem
from gpu_mem_track import MemTracker
from run_gen import KeepKBestModel
import math

matplotlib.use('Agg')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def distill_loss_seq_tree(args, teacher_token_logits, student_token_logits, teacher_token_mask, student_target_mask,
                          teacher_token, target_ids, match_index,
                          match_index_origin):  # 只能拉token的logits, match_index使用情况和distill_loss_tree_seq不一样
    if teacher_token_logits.shape[0] == 0:
        loss = torch.tensor(0.0).to(teacher_token_logits.device)
        return loss
    teacher_valid_mask = teacher_token_mask & (teacher_token != 2)
    student_valid_mask = student_target_mask & (target_ids != 2)
    for index, (match_index_i, match_index_origin_i) in enumerate(zip(match_index, match_index_origin)):
        student_head, student_tail, teacher_head, teacher_tail = match_index_i
        student_head_origin, student_tail_origin, teacher_head_origin, teacher_tail_origin = match_index_origin_i
        teacher_valid_mask[index, :teacher_head_origin] = False
        teacher_valid_mask[index, teacher_tail_origin:] = False
        student_valid_mask[index, :student_head] = False
        student_valid_mask[index, student_tail:] = False
        if student_tail == args.max_target_length:
            teacher_valid_mask[index,
            (teacher_tail_origin - (teacher_valid_mask[index].sum() - student_valid_mask[index].sum())):] = False
        assert teacher_token[index, teacher_valid_mask[index]].equal(target_ids[index, student_valid_mask[index]])
    # 丢弃student_token_logits最后一个维度的最后两个元素
    student_token_logits = student_token_logits[..., :-2]
    # teacher_token_logits在最后一个维度扩展两个0，分别对应<seq>和<tree>
    # teacher_token_logits = torch.cat([teacher_token_logits, torch.zeros(teacher_token_logits.size(0), 2, teacher_token_logits.size(-1)).to(teacher_token_logits.device)], dim=-2)
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_token_logits[student_valid_mask], dim=-1),
        torch.nn.functional.softmax(teacher_token_logits[teacher_valid_mask], dim=-1),
        reduction='none'
    )  # valid_token_num
    loss = loss.sum(-1)
    mask = torch.gather(torch.nn.functional.softmax(teacher_token_logits[teacher_valid_mask], dim=-1), 1,
                        teacher_token[teacher_valid_mask].unsqueeze(-1)) > args.distill_threshold
    loss = loss * mask.squeeze(-1)
    loss = loss.sum() / mask.sum()

    return loss


def distill_loss_tree_seq(args, teacher_token_logits, student_token_logits, teacher_token_mask, student_target_mask,
                          teacher_token, target_ids, match_index, match_index_origin):  # 只能拉token的logits
    if teacher_token_logits.shape[0] == 0:
        return torch.tensor(0.0).to(teacher_token_logits.device)
    teacher_valid_mask = teacher_token_mask & (teacher_token != 2)
    student_valid_mask = student_target_mask & (target_ids != 2)
    for index, (match_index_i, match_index_origin_i) in enumerate(zip(match_index, match_index_origin)):
        teacher_head, teacher_tail, student_head, student_tail = match_index_i
        teacher_head_origin, teacher_tail_origin, student_head_origin, student_tail_origin = match_index_origin_i
        teacher_valid_mask[index, :teacher_head_origin] = False
        teacher_valid_mask[index, teacher_tail_origin:] = False
        student_valid_mask[index, :student_head] = False
        student_valid_mask[index, student_tail:] = False
        if student_tail == args.max_target_length:
            teacher_valid_mask[index,
            (teacher_tail_origin - (teacher_valid_mask[index].sum() - student_valid_mask[index].sum())):] = False
        assert teacher_token[index, teacher_valid_mask[index]].equal(target_ids[index, student_valid_mask[index]])
    student_token_logits = student_token_logits[..., :-2]
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_token_logits[student_valid_mask], dim=-1),
        torch.nn.functional.softmax(teacher_token_logits[teacher_valid_mask], dim=-1),
        reduction='none'
    )  # valid_token_num
    loss = loss.sum(-1)
    mask = torch.gather(torch.nn.functional.softmax(teacher_token_logits[teacher_valid_mask], dim=-1), 1,
                        teacher_token[teacher_valid_mask].unsqueeze(-1)) > args.distill_threshold
    loss = loss * mask.squeeze(-1)
    loss = loss.sum() / mask.sum()

    return loss


def distill_loss_tree_tree(args, teacher_rule_logits, teacher_token_logits, student_rule_logits, student_token_logits,
                           rule_mask, token_mask, rule_ids, target_ids):
    if teacher_rule_logits.shape[0] == 0:
        return torch.tensor(0.0).to(teacher_rule_logits.device)
    # rule loss
    student_rule_logits = student_rule_logits[:, 1:, :]
    teacher_rule_logits = teacher_rule_logits[:, :-1]
    rule_mask = rule_mask[:, :-1]
    rule_ids = rule_ids[:, :-1]
    rule_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_rule_logits[rule_mask], dim=-1),
        torch.nn.functional.softmax(teacher_rule_logits[rule_mask], dim=-1),
        reduction='none'
    )  # valid_token_num
    rule_loss = rule_loss.sum(-1)
    mask = torch.gather(torch.nn.functional.softmax(teacher_rule_logits[rule_mask], dim=-1), 1,
                        rule_ids[rule_mask].unsqueeze(-1)) > args.distill_threshold
    rule_loss = rule_loss * mask.squeeze(-1)
    rule_loss = rule_loss.sum() / mask.sum()

    # token loss
    student_token_logits = student_token_logits[:, 1:, :-2]
    teacher_token_logits = teacher_token_logits[:, :-1]
    token_mask = token_mask[:, :-1]
    target_ids = target_ids[:, :-1]
    token_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_token_logits[token_mask], dim=-1),
        torch.nn.functional.softmax(teacher_token_logits[token_mask], dim=-1),
        reduction='none'
    )  # valid_token_num
    token_loss = token_loss.sum(-1)
    mask = torch.gather(torch.nn.functional.softmax(teacher_token_logits[token_mask], dim=-1), 1,
                        target_ids[token_mask].unsqueeze(-1)) > args.distill_threshold
    token_loss = token_loss * mask.squeeze(-1)
    token_loss = token_loss.sum() / mask.sum()

    loss = token_loss + rule_loss
    return loss


def distill_loss_seq_seq(args, teacher_token_logits, student_token_logits, target_ids, token_mask):  # 只能拉token的logits
    if teacher_token_logits.shape[0] == 0:
        return torch.tensor(0.0).to(teacher_token_logits.device)
    teacher_token_logits = teacher_token_logits[:, :-1, :]
    student_token_logits = student_token_logits[:, 1:, :-2]
    token_mask = token_mask[:, :-1]
    target_ids = target_ids[:, :-1]
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_token_logits, dim=-1),
        torch.nn.functional.softmax(teacher_token_logits, dim=-1),
        reduction='none'
    )  # valid_token_num
    loss = loss.sum(-1)
    mask = torch.gather(torch.nn.functional.softmax(teacher_token_logits, dim=-1), 2,
                        target_ids.unsqueeze(-1)) > args.distill_threshold
    mask = mask.squeeze(-1) & token_mask
    loss = loss * mask
    loss = loss.sum() / mask.sum()

    return loss


def distill_opts(parser):
    parser.add_argument("--tree_path", default=None, type=str, help="The path to the best tree model.")
    parser.add_argument("--seq_path", default=None, type=str, help="The path to the best seq model.")
    parser.add_argument("--best_both_path", default=None, type=str, help="The path to the best both model.")
    parser.add_argument("--distill_type", type=str, choices=['both', 'cross', 'homogeneous', 'tree'])
    parser.add_argument("--distill_lambda", type=float, default=0.1)
    parser.add_argument("--distill_threshold", type=float, default=0.5)
    parser.add_argument('--decode_label_path', type=str, default=None, help='path to decoder label')


def main():
    parser = argparse.ArgumentParser()
    distill_opts(parser)
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

    # grammar = ASDLGrammar.from_text(open('asdl/lang/java/java_asdl.txt').read(), 'program')
    grammar_txt = 'asdl/lang/java/java_asdl.txt'
    if args.task == 'translate' and args.sub_task == 'cs-java':
        grammar_txt = 'asdl/lang/java/java_asdl_translate.txt'
        src_grammar_txt = 'asdl/lang/java/cs_asdl_translate.txt'
        src_grammar = ASDLGrammar.from_text(open(src_grammar_txt).read(), 'program')
        src_transition_system = JavaTransitionSystem(src_grammar)
    else:
        src_transition_system = None
    grammar = ASDLGrammar.from_text(open(grammar_txt).read(), 'program')
    logger.info('grammar path: {}'.format(grammar_txt))
    logger.info('grammar length: {}'.format(len(grammar)))
    transition_system = JavaTransitionSystem(grammar)
    # load one TreeCodeT5 models and one CodeT5 model
    teacher_tree_model = TreeCodeT5(args, vocab=None, transition_system=transition_system)
    if args.tree_path is not None and os.path.exists(args.tree_path):
        logger.info("Reload teacher tree model from {}".format(args.tree_path))
        state_dict = torch.load(args.tree_path)
        if not args.source_ast:
            unmatched_key = 'src_production_embed.weight'
            if unmatched_key in state_dict:
                del state_dict[unmatched_key]
        teacher_tree_model.load_state_dict(state_dict, strict=False)
    teacher_tree_model.to(args.device)

    config, teacher_seq_model, tokenizer = build_or_load_gen_model(args)
    if args.seq_path is not None and os.path.exists(args.seq_path):
        logger.info("Reload teacher seq model from {}".format(args.seq_path))
        teacher_seq_model.load_state_dict(torch.load(args.seq_path))
    teacher_seq_model.to(args.device)

    # init student model
    student_model = TreeSeqCodeT5(args, vocab=None, transition_system=transition_system)
    tokenizer = student_model.tokenizer
    data_load_function = load_and_cache_gen_tree_data

    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        logger.info("Reload student_model from {}".format(args.load_model_path))
        student_model.load_state_dict(torch.load(args.load_model_path), strict=False)
    student_model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        student_model = torch.nn.DataParallel(student_model)
    pool = multiprocessing.Pool(1 if args.debug else args.cpu_cont)  # args.cpu_cont
    args.train_filename, args.dev_filename, test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if args.test_filename is None:
        args.test_filename = test_filename
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train and (args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path)):
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = data_load_function(args, args.train_filename, pool, tokenizer, 'train',
                                                        multitask=True, distill=True, decode_label_path=args.decode_label_path)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        if args.debug:
            train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size,
                                          num_workers=4, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                          num_workers=4, pin_memory=True)
            
        if args.tune_on_label:
            # 冻结分类器之外的所有参数
            for name, param in student_model.named_parameters():
                if 'decode_classifier' not in name:
                    param.requires_grad = False

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
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
        global_step, best_bleu_em_both, best_bleu_em_seq, best_bleu_em_tree, best_ppl = 0, -1, -1, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        kbModels_seq = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-bleu-seq'))
        kbModels_tree = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-bleu-tree'))
        kbModels_both = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-best-bleu-both'))
        klModels = KeepKBestModel(5, os.path.join(args.output_dir, 'checkpoint-last'))

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            student_model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                if args.decode_label_path is None:
                    source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row, match_index, \
                    match_index_origin, target_ids_origin, app_rule_idx_row_origin, app_rule_mask_row_origin, token_row_origin, gen_token_mask_row_origin = batch
                    first_token_loss_mask = None
                else:
                    source_ids, target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row, match_index, \
                    match_index_origin, target_ids_origin, app_rule_idx_row_origin, app_rule_mask_row_origin, token_row_origin, gen_token_mask_row_origin, \
                    first_token_loss_mask = batch
                tree_labels = (app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
                tree_labels_origin = (
                app_rule_idx_row_origin, app_rule_mask_row_origin, token_row_origin, gen_token_mask_row_origin)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                tree_target_mask = (app_rule_mask_row + gen_token_mask_row).bool()
                tree_target_mask_origin = (app_rule_mask_row_origin + gen_token_mask_row_origin).bool()
                seq_target_mask = target_ids.ne(tokenizer.pad_token_id)
                seq_target_mask_origin = target_ids_origin.ne(tokenizer.pad_token_id)

                # target_ids[:, 0] = tokenizer.convert_tokens_to_ids("<tree>")
                decode_type_label = target_ids[:, 0] == tokenizer.convert_tokens_to_ids(
                    "<seq>")  # true代表seq，false代表tree
                student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask,
                                                seq_labels=target_ids, tree_labels=tree_labels,
                                                seq_decoder_attention_mask=seq_target_mask,
                                                tree_decoder_attention_mask=tree_target_mask,
                                                decode_type_label=decode_type_label,
                                                first_token_loss_mask=first_token_loss_mask)
                student_loss, student_loss_seq, student_loss_tree = student_outputs[0:3]
                student_token_logits = student_outputs[3]
                student_rule_logits = student_outputs[4]
                # if args.distill_type == 'both':
                #     with torch.no_grad():
                #         teacher_both_outputs = teacher_both_model(input_ids=source_ids, attention_mask=source_mask,
                #                                                   seq_labels=target_ids, tree_labels=tree_labels,
                #                                                   seq_decoder_attention_mask=seq_target_mask,
                #                                                   tree_decoder_attention_mask=tree_target_mask,
                #                                                   decode_type_label=decode_type_label)
                #     teacher_both_token_logits = teacher_both_outputs[3]
                #     teacher_both_rule_logits = teacher_both_outputs[4]
                #     distill_loss = distill_loss_seq_seq(args, teacher_both_token_logits[decode_type_label],
                #                                         student_token_logits[decode_type_label],
                #                                         target_ids[decode_type_label],
                #                                         seq_target_mask[decode_type_label]) + \
                #                    distill_loss_tree_tree(args, teacher_both_rule_logits[~decode_type_label],
                #                                           teacher_both_token_logits[~decode_type_label],
                #                                           student_rule_logits[~decode_type_label],
                #                                           student_token_logits[~decode_type_label],
                #                                           app_rule_mask_row.bool()[~decode_type_label],
                #                                           gen_token_mask_row.bool()[~decode_type_label],
                #                                           app_rule_idx_row[~decode_type_label],
                #                                           token_row[~decode_type_label])
                # elif args.distill_type == 'cross' or args.distill_type == 'homogeneous':
                with torch.no_grad():
                    # 这里两个教师模型使用的数据不应该包括<seq><tree>, 否则会超过词表的限制
                    teacher_tree_outputs = teacher_tree_model(input_ids=source_ids, attention_mask=source_mask,
                                                                labels=tree_labels_origin,
                                                                decoder_attention_mask=tree_target_mask_origin)
                    teacher_tree_token_logits = teacher_tree_outputs[1]
                    teacher_tree_rule_logits = teacher_tree_outputs[2]
                    teacher_seq_outputs = teacher_seq_model(input_ids=source_ids, attention_mask=source_mask,
                                                            labels=target_ids_origin,
                                                            decoder_attention_mask=seq_target_mask_origin)
                    teacher_seq_token_logits = teacher_seq_outputs.logits
                if args.distill_type == 'cross':
                    distill_loss = distill_loss_seq_tree(args, teacher_seq_token_logits[~decode_type_label],
                                                            student_token_logits[~decode_type_label],
                                                            seq_target_mask_origin[~decode_type_label],
                                                            gen_token_mask_row[~decode_type_label].bool(),
                                                            target_ids_origin[~decode_type_label],
                                                            token_row[~decode_type_label],
                                                            match_index[~decode_type_label],
                                                            match_index_origin[~decode_type_label]) + \
                                    distill_loss_tree_seq(args, teacher_tree_token_logits[decode_type_label],
                                                            student_token_logits[decode_type_label],
                                                            gen_token_mask_row_origin[decode_type_label].bool(),
                                                            seq_target_mask[decode_type_label],
                                                            token_row_origin[decode_type_label],
                                                            target_ids[decode_type_label],
                                                            match_index[decode_type_label],
                                                            match_index_origin[decode_type_label])
                elif args.distill_type == 'homogeneous':
                    distill_loss = distill_loss_seq_seq(args, teacher_seq_token_logits[decode_type_label],
                                                        student_token_logits[decode_type_label],
                                                        target_ids_origin[decode_type_label],
                                                        seq_target_mask_origin[decode_type_label]) + \
                                    distill_loss_tree_tree(args, teacher_tree_rule_logits[~decode_type_label],
                                                            teacher_tree_token_logits[~decode_type_label],
                                                            student_rule_logits[~decode_type_label],
                                                            student_token_logits[~decode_type_label],
                                                            app_rule_mask_row_origin.bool()[~decode_type_label],
                                                            gen_token_mask_row_origin.bool()[~decode_type_label],
                                                            app_rule_idx_row_origin[~decode_type_label],
                                                            token_row_origin[~decode_type_label], )
                elif args.distill_type == 'both':
                    distill_loss = distill_loss_seq_tree(args, teacher_seq_token_logits[~decode_type_label],
                                                            student_token_logits[~decode_type_label],
                                                            seq_target_mask_origin[~decode_type_label],
                                                            gen_token_mask_row[~decode_type_label].bool(),
                                                            target_ids_origin[~decode_type_label],
                                                            token_row[~decode_type_label],
                                                            match_index[~decode_type_label],
                                                            match_index_origin[~decode_type_label]) + \
                                    distill_loss_tree_seq(args, teacher_tree_token_logits[decode_type_label],
                                                            student_token_logits[decode_type_label],
                                                            gen_token_mask_row_origin[decode_type_label].bool(),
                                                            seq_target_mask[decode_type_label],
                                                            token_row_origin[decode_type_label],
                                                            target_ids[decode_type_label],
                                                            match_index[decode_type_label],
                                                            match_index_origin[decode_type_label]) + \
                                    distill_loss_seq_seq(args, teacher_seq_token_logits[decode_type_label],
                                                        student_token_logits[decode_type_label],
                                                        target_ids_origin[decode_type_label],
                                                        seq_target_mask_origin[decode_type_label]) + \
                                    distill_loss_tree_tree(args, teacher_tree_rule_logits[~decode_type_label],
                                                            teacher_tree_token_logits[~decode_type_label],
                                                            student_rule_logits[~decode_type_label],
                                                            student_token_logits[~decode_type_label],
                                                            app_rule_mask_row_origin.bool()[~decode_type_label],
                                                            gen_token_mask_row_origin.bool()[~decode_type_label],
                                                            app_rule_idx_row_origin[~decode_type_label],
                                                            token_row_origin[~decode_type_label], )
                elif args.distill_type == 'tree':
                    distill_loss = distill_loss_tree_seq(args, teacher_tree_token_logits[decode_type_label],
                                                            student_token_logits[decode_type_label],
                                                            gen_token_mask_row_origin[decode_type_label].bool(),
                                                            seq_target_mask[decode_type_label],
                                                            token_row_origin[decode_type_label],
                                                            target_ids[decode_type_label],
                                                            match_index[decode_type_label],
                                                            match_index_origin[decode_type_label]) + \
                                    distill_loss_seq_seq(args, teacher_seq_token_logits[decode_type_label],
                                                        student_token_logits[decode_type_label],
                                                        target_ids_origin[decode_type_label],
                                                        seq_target_mask_origin[decode_type_label]) + \
                                    distill_loss_tree_tree(args, teacher_tree_rule_logits[~decode_type_label],
                                                            teacher_tree_token_logits[~decode_type_label],
                                                            student_rule_logits[~decode_type_label],
                                                            student_token_logits[~decode_type_label],
                                                            app_rule_mask_row_origin.bool()[~decode_type_label],
                                                            gen_token_mask_row_origin.bool()[~decode_type_label],
                                                            app_rule_idx_row_origin[~decode_type_label],
                                                            token_row_origin[~decode_type_label], )
                loss = (1 - args.distill_lambda) * student_loss + args.distill_lambda * distill_loss
                if args.tune_on_label:
                    loss = student_loss

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
                    bar.set_description(
                        "[{}] Train loss {}, student loss {}, distill_loss {}".format(cur_epoch, round(loss.item(), 3),
                                                                                      round(student_loss.item(), 3),
                                                                                      round(distill_loss.item(), 3)))

                if args.no_train:
                    break

            if args.do_eval:
                # Eval student_model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                  multitask=True)
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl, eval_seq_ppl, eval_tree_ppl = eval_ppl_epoch(args, eval_data, eval_examples, student_model,
                                                                       tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl,
                          'eval_seq_ppl': eval_seq_ppl, 'eval_tree_ppl': eval_tree_ppl}
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
                    logger.info("Save the last student_model into %s", output_model_file)
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
                        logger.info("Save the best ppl student_model into %s", output_model_file)
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
                    eval_examples, eval_data = data_load_function(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                  only_src=True, is_sample=True)  # , sample_number=50
                    decode_types = ['seq', 'tree']
                    dev_bleu_em_both = 0
                    if not args.tune_on_label:
                        for decode_type in decode_types:
                            if decode_type == 'seq':
                                best_bleu_em = best_bleu_em_seq
                                kbModels = kbModels_seq
                            else:
                                best_bleu_em = best_bleu_em_tree
                                kbModels = kbModels_tree
                            result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, 'dev',
                                                    'e%d' % cur_epoch, decode_type=decode_type)
                            if args.task in ['summarize']:
                                dev_bleu_em = dev_bleu
                            else:
                                dev_bleu, dev_em = result['bleu'], result['em']
                                dev_bleu_em = dev_bleu + dev_em
                                dev_bleu_em_both += dev_bleu_em
                            if args.data_num == -1:
                                tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                                # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                            if dev_bleu_em > best_bleu_em:
                                logger.info("  [%d][%s] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                            cur_epoch, decode_type, dev_bleu_em, dev_bleu, dev_em)
                                logger.info("  " + "*" * 20)
                                best_bleu_em = dev_bleu_em
                                fa.write("[%d][%s] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                                    cur_epoch, decode_type, best_bleu_em, dev_bleu, dev_em))
                                # Save best checkpoint for best bleu
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu-' + decode_type)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                if args.data_num == -1 or args.always_save_model:
                                    model_to_save = student_model.module if hasattr(student_model,
                                                                                    'module') else student_model
                                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), output_model_file)
                                    logger.info("Save the best %s bleu student_model into %s", decode_type,
                                                output_model_file)
                            kbModels.add(student_model, dev_bleu_em, cur_epoch)  # 保存到对应解码器的kbModels
                    else:
                        result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, 'dev', 'e%d' % cur_epoch)
                        dev_bleu_both, dev_em_both = result['bleu'], result['em']
                        if args.task in ['summarize']:
                            dev_bleu_em_both = dev_bleu_both
                        else:
                            dev_bleu_em_both = dev_bleu_both + dev_em_both
                        if args.data_num == -1:
                            tb_writer.add_scalar('dev_bleu_em', dev_bleu_em_both, cur_epoch)
                    if dev_bleu_em_both > best_bleu_em_both:
                        best_bleu_em_both = dev_bleu_em_both
                        not_bleu_em_inc_cnt = 0
                        if not args.tune_on_label:
                            logger.info("  [%d] Best bleu+em: %.2f (seq: %.2f, tree: %.2f)",
                                            cur_epoch, best_bleu_em_both, best_bleu_em_both - (dev_bleu+dev_em), dev_bleu+dev_em)
                            logger.info("  " + "*" * 20)
                            fa.write("[%d] Best bleu+em changed into %.2f (seq: %.2f, tree: %.2f)\n" % (
                                cur_epoch, best_bleu_em_both, best_bleu_em_both - (dev_bleu+dev_em), dev_bleu+dev_em))
                        else:
                            logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, best_bleu_em_both, dev_bleu_both, dev_em_both)
                            logger.info("  " + "*" * 20)
                            fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em_both, dev_bleu_both, dev_em_both))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu-both')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu student_model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f\n" % (
                                cur_epoch, best_bleu_em_both, not_bleu_em_inc_cnt, dev_bleu_em_both))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
                    kbModels_both.add(student_model, dev_bleu_em_both, cur_epoch)
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        criteria_list = ['best-bleu-both'] if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best-bleu-both/pytorch_model.bin')) else ['best-ppl']
        if args.avg_checkpoint_path is None or not os.path.exists(args.avg_checkpoint_path):
            criteria_list.append('average.bin')
        for criteria in criteria_list:
            if criteria == 'average.bin': # test after train mode
                file = os.path.join(args.output_dir, 'checkpoint-best-bleu-both/average.bin')
                if not os.path.exists(file):
                    status = os.system('python average_checkpoints.py --path '+os.path.join(args.output_dir, 'checkpoint-best-bleu-both')+' --output '+os.path.join(args.output_dir, 'checkpoint-best-bleu-both'))
                    if status == 0:
                        logger.info("success generate average.bin")
                    else:  # if happen in pycharm run/debug, don't care, it will work well in the server
                        logger.error("fail to generate average.bin")
            else:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if args.avg_checkpoint_path is not None and os.path.exists(args.avg_checkpoint_path):
                file = args.avg_checkpoint_path  # test-only mode
            logger.info("Reload student_model from {}".format(file))
            student_model.load_state_dict(torch.load(file))
            eval_examples, eval_data = data_load_function(args, args.test_filename, pool, tokenizer, args.test_split_tag,
                                                          only_src=True, is_sample=False)
            if not args.tune_on_label or args.decode_label_path is not None:
                for decode_type in ['tree']:
                    result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, args.test_split_tag, criteria, decode_type=decode_type)
                    test_bleu, test_em = result['bleu'], result['em']
                    test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                    result_str = "[%s][%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, decode_type, test_bleu, test_em, test_codebleu)
                    logger.info(result_str)
                    fa.write(result_str)
                    if args.res_fn:
                        with open(args.res_fn, 'a+') as f:
                            f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                            f.write(result_str)
            else:
                result = eval_bleu_epoch(args, eval_data, eval_examples, student_model, tokenizer, args.test_split_tag, criteria)
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
                break  # test-only mode
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == '__main__':
    main()
