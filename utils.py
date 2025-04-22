from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *
import pickle
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import torch.nn as nn
from evaluator.bleu import compute_bleu

logger = logging.getLogger(__name__)


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    if args.debug:
        data_tag = '_debug'
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    if args.debug:
        filename = '/'.join(filename.split('/')[:-1]) + '/debug_' + filename.split('/')[-1]
    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_conala_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    def read_conala_examples(filename, split_tag):
        """Read examples from filename."""
        examples = []

        with open(filename) as f:
            if 'mined' in filename:
                # read jsonl file, each line is a json object
                data = f.readlines()
                json_list = []
                for idx, x in enumerate(data):
                    json_list.append(json.loads(x))
            else:
                json_list = json.load(f)
                if split_tag == 'dev':
                    json_list = json_list[-500:]
                elif split_tag == 'train':
                    json_list = json_list[:-500]
            for idx, x in enumerate(json_list):
                if split_tag == 'train' and 'mined' in filename:
                    if x['intent']:
                        e = Example(
                                idx=idx,
                                source=x['intent'].strip(),
                                target=x["snippet"].strip()
                            )
                        e.target = e.target.replace('\n', ' ')
                        examples.append(e)
                elif split_tag == 'train':
                    # train, use both rewritten_intent, if none, use intent
                    e = Example(
                            idx=idx,
                            source=x['rewritten_intent'].strip() if x['rewritten_intent'] else x['intent'].strip(),
                            target=x["snippet"].strip()
                        )
                    e.target = e.target.replace('\n', ' ')
                    examples.append(e)

                else:
                    if not x['rewritten_intent']:
                        continue
                    e = Example(
                            idx=idx,
                            source=x['rewritten_intent'].strip(),
                            target=x["snippet"].strip()
                        )
                    
                    e.target = e.target.replace('\n', ' ')
                    examples.append(e)
        return examples
    
    def read_conala_example_(filename):
        # src = 'nl', tgt = 'cmd', jsonl file
        examples = []
        with open(filename) as f:
            data = f.readlines()
            json_list = []
            for idx, x in enumerate(data):
                json_list.append(json.loads(x))
            for idx, x in enumerate(json_list):
                e = Example(
                        idx=idx,
                        source=x['nl'].strip(),
                        target=x["cmd"].strip()
                    )
                e.target = e.target.replace('\n', ' ')
                examples.append(e)
            
        return examples
    
    def read_conala_example_doc(filename):
        # question, target, ctxs json list
        examples = []
        with open(filename) as f:
            json_list = json.load(f)
            for idx, x in enumerate(json_list):
                e = Example(
                        idx=idx,
                        source=x['question'].strip() + '. ' + x['ctxs'][0]['title'].strip() + '. ' + x['ctxs'][0]['text'].strip(),
                        target=x["target"].strip()
                    )
                e.target = e.target.replace('\n', ' ')
                examples.append(e)
        
        return examples
    
    examples = read_conala_example_doc(filename)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    
    # source, target, url
    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    if split_tag == 'test' or only_src:
        data = TensorDataset(all_source_ids)
    else:
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)
    return examples, data


def load_conala_tree_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    def read_concode_examples(filename, data_num):
        """Read examples from filename."""
        examples = []

        with open(filename) as f:
            json_list = json.load(f)
            for idx, x in enumerate(json_list):
                e = Example(
                        idx=idx,
                        source=x['rewritten_intent'].strip() if x['rewritten_intent'] else x['intent'].strip(),
                        target=x["snippet"].strip()
                    )
                e.target = e.target.replace('\n', ' ')
                examples.append(e)
                idx += 1
                if idx == data_num:
                    break
        return examples
    examples = read_concode_examples(filename, -1)
    if split_tag == 'dev':
        examples = examples[-500:]
    elif split_tag == 'train':
        examples = examples[:-500]

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    
    # source, target, url
    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    if split_tag == 'test' or only_src:
        data = TensorDataset(all_source_ids)
    else:
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        if split_tag == 'train' or split_tag == 'dev':
            tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/train_conala.bin"
        else:
            tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/test_conala.bin"
        print('tree_filename:', tree_filename)
        tree_labels = pickle.load(open(tree_filename, 'rb'))

        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        # app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row
        app_rule_idx_row = torch.tensor([f[0][:args.max_target_length] + [0] * (args.max_target_length - len(f[0][:args.max_target_length]))
                                                        for f in tree_labels], dtype=torch.long)
        app_rule_mask_row = torch.tensor([f[1][:args.max_target_length] + [0] * (args.max_target_length - len(f[1][:args.max_target_length]))
                                                        for f in tree_labels], dtype=torch.long)
        token_row = torch.tensor([f[2][:args.max_target_length] + [0] * (args.max_target_length - len(f[2][:args.max_target_length]))
                                                        for f in tree_labels], dtype=torch.long)
        gen_token_mask_row = torch.tensor([f[3][:args.max_target_length] + [0] * (args.max_target_length - len(f[3][:args.max_target_length]))
                                                        for f in tree_labels], dtype=torch.long)
        
        if split_tag == 'train':
            app_rule_idx_row = app_rule_idx_row[:-500]
            app_rule_mask_row = app_rule_mask_row[:-500]
            token_row = token_row[:-500]
            gen_token_mask_row = gen_token_mask_row[:-500]
        elif split_tag == 'dev':
            app_rule_idx_row = app_rule_idx_row[-500:]
            app_rule_mask_row = app_rule_mask_row[-500:]
            token_row = token_row[-500:]
            gen_token_mask_row = gen_token_mask_row[-500:]
        data = TensorDataset(all_source_ids, all_target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
    return examples, data


def load_and_cache_gen_tree_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False, sample_number=0, distill=False, multitask=False, decode_label_path=None):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    if args.debug:
        data_tag = '_debug'
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    if decode_label_path is not None and split_tag != 'test': 
        # decode_label_path = /home/sata/sly/CG/bin/multitask_train_1_decode_label.pkl, cache_fn = .../train_all.pt->.../multitask_train_all.pt
        # decode_label_path = /home/sata/sly/CG/bin/multitask_distill_cross_train_1_decode_label.pkl, cache_fn = .../train_all.pt->.../multitask_distill_cross_train_all.pt
        mode = '_'.join(decode_label_path.split('/')[-1].split('1')[0].split('_')[:-2])
        cache_fn = '/'.join(cache_fn.split('/')[:-1]) + '/' + mode + '_'+ cache_fn.split('/')[-1]
    print(cache_fn, filename)

    if args.debug:
        filename = '/'.join(filename.split('/')[:-1]) + '/debug_' + filename.split('/')[-1]
    examples = read_examples(filename, args.data_num, args.task)
    # if args.debug:
    #     examples = examples[:10]
        # print(examples)

    # is_sample = True
    if is_sample:
        examples = random.sample(examples, min(5000 if not sample_number else sample_number, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            if args.task == 'concode':
                tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + ".bin"
            elif args.task == 'translate' and args.sub_task == 'cs-java':
                tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_translate_cs-java_java.bin"
            print('tree_filename:', tree_filename)
            tree_labels = pickle.load(open(tree_filename, 'rb'))

            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            # app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row
            app_rule_idx_row = torch.tensor([f[0][:args.max_target_length] + [0] * (args.max_target_length - len(f[0][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            app_rule_mask_row = torch.tensor([f[1][:args.max_target_length] + [0] * (args.max_target_length - len(f[1][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            token_row = torch.tensor([f[2][:args.max_target_length] + [0] * (args.max_target_length - len(f[2][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            gen_token_mask_row = torch.tensor([f[3][:args.max_target_length] + [0] * (args.max_target_length - len(f[3][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            if multitask and decode_label_path is None:
                # 整体右移一位，并将第一个token替换为<seq>
                all_target_ids_seq = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]] * all_target_ids.shape[0], dtype=torch.long), all_target_ids[:, :-1]], dim=1)
                app_rule_idx_row_seq = torch.cat([torch.tensor([[0]] * app_rule_idx_row.shape[0], dtype=torch.long), app_rule_idx_row[:, :-1]], dim=1)
                app_rule_mask_row_seq = torch.cat([torch.tensor([[0]] * app_rule_mask_row.shape[0], dtype=torch.long), app_rule_mask_row[:, :-1]], dim=1)
                token_row_seq = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]] * token_row.shape[0], dtype=torch.long), token_row[:, :-1]], dim=1)
                gen_token_mask_row_seq = torch.cat([torch.tensor([[1]] * gen_token_mask_row.shape[0], dtype=torch.long), gen_token_mask_row[:, :-1]], dim=1)
                # 整体右移一位，并将第一个token替换为<tree>
                all_target_ids_tree = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]] * all_target_ids.shape[0], dtype=torch.long), all_target_ids[:, :-1]], dim=1)
                app_rule_idx_row_tree = torch.cat([torch.tensor([[0]] * app_rule_idx_row.shape[0], dtype=torch.long), app_rule_idx_row[:, :-1]], dim=1)
                app_rule_mask_row_tree = torch.cat([torch.tensor([[0]] * app_rule_mask_row.shape[0], dtype=torch.long), app_rule_mask_row[:, :-1]], dim=1)
                token_row_tree = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]] * token_row.shape[0], dtype=torch.long), token_row[:, :-1]], dim=1)
                gen_token_mask_row_tree = torch.cat([torch.tensor([[1]] * gen_token_mask_row.shape[0], dtype=torch.long), gen_token_mask_row[:, :-1]], dim=1)
                if distill:
                    # clone原来的target_ids、app_rule_idx_row、app_rule_mask_row、token_row、gen_token_mask_row
                    target_ids_origin = torch.cat([all_target_ids.clone(), all_target_ids.clone()], dim=0)
                    app_rule_idx_row_origin = torch.cat([app_rule_idx_row.clone(), app_rule_idx_row.clone()], dim=0)
                    app_rule_mask_row_origin = torch.cat([app_rule_mask_row.clone(), app_rule_mask_row.clone()], dim=0)
                    token_row_origin = torch.cat([token_row.clone(), token_row.clone()], dim=0)
                    gen_token_mask_row_origin = torch.cat([gen_token_mask_row.clone(), gen_token_mask_row.clone()], dim=0)
                # 更新原数据
                all_source_ids = torch.cat([all_source_ids, all_source_ids], dim=0)
                all_target_ids = torch.cat([all_target_ids_seq, all_target_ids_tree], dim=0)
                app_rule_idx_row = torch.cat([app_rule_idx_row_seq, app_rule_idx_row_tree], dim=0)
                app_rule_mask_row = torch.cat([app_rule_mask_row_seq, app_rule_mask_row_tree], dim=0)
                token_row = torch.cat([token_row_seq, token_row_tree], dim=0)
                gen_token_mask_row = torch.cat([gen_token_mask_row_seq, gen_token_mask_row_tree], dim=0)
            elif multitask and decode_label_path is not None:
                # load decode_label
                decode_label = pickle.load(open(decode_label_path, 'rb'))
                if args.debug:
                    decode_label = decode_label[:100]
                # 根据decode_label提前确定训练集样本数量
                dataset_size = 0
                for value in decode_label:
                    if value == 0 or value == 1:
                        dataset_size += 1
                    else:
                        dataset_size += 2
                logger.info('dataset_size: {}'.format(dataset_size))
                labeled_source_ids = torch.empty(((dataset_size,) + all_source_ids.shape[1:]), dtype=torch.long)
                labeled_target_ids = torch.empty(((dataset_size,) + all_target_ids.shape[1:]), dtype=torch.long)
                labeled_app_rule_idx_row = torch.empty(((dataset_size,) + app_rule_idx_row.shape[1:]), dtype=torch.long)
                labeled_app_rule_mask_row = torch.empty(((dataset_size,) + app_rule_mask_row.shape[1:]), dtype=torch.long)
                labeled_token_row = torch.empty(((dataset_size,) + token_row.shape[1:]), dtype=torch.long)
                labeled_gen_token_mask_row = torch.empty(((dataset_size,) + gen_token_mask_row.shape[1:]), dtype=torch.long)
                first_token_loss_mask = []
                # 遍历decode_label的值，0-<seq>, 1-<tree>, 2-<seq>&<tree>
                target_ids_origin = torch.empty(((dataset_size,) + all_target_ids.shape[1:]), dtype=torch.long)
                app_rule_idx_row_origin = torch.empty(((dataset_size,) + app_rule_idx_row.shape[1:]), dtype=torch.long)
                app_rule_mask_row_origin = torch.empty(((dataset_size,) + app_rule_mask_row.shape[1:]), dtype=torch.long)
                token_row_origin = torch.empty(((dataset_size,) + token_row.shape[1:]), dtype=torch.long)
                gen_token_mask_row_origin = torch.empty(((dataset_size,) + gen_token_mask_row.shape[1:]), dtype=torch.long)
                counter = 0
                for index, value in enumerate(tqdm(decode_label)):
                    if value == 0 or value == 2:
                        if value == 2:
                            # 不计算第一个token的loss, False
                            first_token_loss_mask.append(False)
                        else:
                            first_token_loss_mask.append(True)
                        # 右移一位，并将第一个token替换为<seq>
                        # labeled_source_ids = torch.cat([labeled_source_ids, all_source_ids[index].unsqueeze(0)], dim=0) if labeled_source_ids is not None else all_source_ids[index].unsqueeze(0)
                        labeled_source_ids[counter] = all_source_ids[index]
                        
                        index_target_ids = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]], dtype=torch.long), all_target_ids[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_target_ids = torch.cat([labeled_target_ids, index_target_ids], dim=0) if labeled_target_ids is not None else index_target_ids
                        labeled_target_ids[counter] = index_target_ids
                        
                        index_app_rule_idx_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_idx_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_idx_row = torch.cat([labeled_app_rule_idx_row, index_app_rule_idx_row], dim=0) if labeled_app_rule_idx_row is not None else index_app_rule_idx_row
                        labeled_app_rule_idx_row[counter] = index_app_rule_idx_row
                        
                        index_app_rule_mask_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_mask_row = torch.cat([labeled_app_rule_mask_row, index_app_rule_mask_row], dim=0) if labeled_app_rule_mask_row is not None else index_app_rule_mask_row
                        labeled_app_rule_mask_row[counter] = index_app_rule_mask_row
                        
                        index_token_row = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]], dtype=torch.long), token_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_token_row = torch.cat([labeled_token_row, index_token_row], dim=0) if labeled_token_row is not None else index_token_row
                        labeled_token_row[counter] = index_token_row

                        index_gen_token_mask_row = torch.cat([torch.tensor([[1]], dtype=torch.long), gen_token_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_gen_token_mask_row = torch.cat([labeled_gen_token_mask_row, index_gen_token_mask_row], dim=0) if labeled_gen_token_mask_row is not None else index_gen_token_mask_row
                        labeled_gen_token_mask_row[counter] = index_gen_token_mask_row
                        if distill:
                            # target_ids_origin = torch.cat([target_ids_origin, all_target_ids[index].unsqueeze(0)], dim=0) if target_ids_origin is not None else all_target_ids[index].unsqueeze(0)
                            target_ids_origin[counter] = all_target_ids[index]
                            # app_rule_idx_row_origin = torch.cat([app_rule_idx_row_origin, app_rule_idx_row[index].unsqueeze(0)], dim=0) if app_rule_idx_row_origin is not None else app_rule_idx_row[index].unsqueeze(0)
                            app_rule_idx_row_origin[counter] = app_rule_idx_row[index]
                            # app_rule_mask_row_origin = torch.cat([app_rule_mask_row_origin, app_rule_mask_row[index].unsqueeze(0)], dim=0) if app_rule_mask_row_origin is not None else app_rule_mask_row[index].unsqueeze(0)
                            app_rule_mask_row_origin[counter] = app_rule_mask_row[index]
                            # token_row_origin = torch.cat([token_row_origin, token_row[index].unsqueeze(0)], dim=0) if token_row_origin is not None else token_row[index].unsqueeze(0)
                            token_row_origin[counter] = token_row[index]
                            # gen_token_mask_row_origin = torch.cat([gen_token_mask_row_origin, gen_token_mask_row[index].unsqueeze(0)], dim=0) if gen_token_mask_row_origin is not None else gen_token_mask_row[index].unsqueeze(0)
                            gen_token_mask_row_origin[counter] = gen_token_mask_row[index]
                        counter += 1
                    if value == 1 or value == 2:
                        if value == 2:
                            # 不计算第一个token的loss, False
                            first_token_loss_mask.append(False)
                        else:
                            first_token_loss_mask.append(True)
                        # 右移一位，并将第一个token替换为<tree>
                        # labeled_source_ids = torch.cat([labeled_source_ids, all_source_ids[index].unsqueeze(0)], dim=0) if labeled_source_ids is not None else all_source_ids[index].unsqueeze(0)
                        labeled_source_ids[counter] = all_source_ids[index]

                        index_target_ids = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]], dtype=torch.long), all_target_ids[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_target_ids = torch.cat([labeled_target_ids, index_target_ids], dim=0) if labeled_target_ids is not None else index_target_ids
                        labeled_target_ids[counter] = index_target_ids

                        index_app_rule_idx_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_idx_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_idx_row = torch.cat([labeled_app_rule_idx_row, index_app_rule_idx_row], dim=0) if labeled_app_rule_idx_row is not None else index_app_rule_idx_row
                        labeled_app_rule_idx_row[counter] = index_app_rule_idx_row

                        index_app_rule_mask_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_mask_row = torch.cat([labeled_app_rule_mask_row, index_app_rule_mask_row], dim=0) if labeled_app_rule_mask_row is not None else index_app_rule_mask_row
                        labeled_app_rule_mask_row[counter] = index_app_rule_mask_row

                        index_token_row = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]], dtype=torch.long), token_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_token_row = torch.cat([labeled_token_row, index_token_row], dim=0) if labeled_token_row is not None else index_token_row
                        labeled_token_row[counter] = index_token_row

                        index_gen_token_mask_row = torch.cat([torch.tensor([[1]], dtype=torch.long), gen_token_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_gen_token_mask_row = torch.cat([labeled_gen_token_mask_row, index_gen_token_mask_row], dim=0) if labeled_gen_token_mask_row is not None else index_gen_token_mask_row
                        labeled_gen_token_mask_row[counter] = index_gen_token_mask_row
                        
                        if distill:
                            # target_ids_origin = torch.cat([target_ids_origin, all_target_ids[index].unsqueeze(0)], dim=0) if target_ids_origin is not None else all_target_ids[index].unsqueeze(0)
                            # app_rule_idx_row_origin = torch.cat([app_rule_idx_row_origin, app_rule_idx_row[index].unsqueeze(0)], dim=0) if app_rule_idx_row_origin is not None else app_rule_idx_row[index].unsqueeze(0)
                            # app_rule_mask_row_origin = torch.cat([app_rule_mask_row_origin, app_rule_mask_row[index].unsqueeze(0)], dim=0) if app_rule_mask_row_origin is not None else app_rule_mask_row[index].unsqueeze(0)
                            # token_row_origin = torch.cat([token_row_origin, token_row[index].unsqueeze(0)], dim=0) if token_row_origin is not None else token_row[index].unsqueeze(0)
                            # gen_token_mask_row_origin = torch.cat([gen_token_mask_row_origin, gen_token_mask_row[index].unsqueeze(0)], dim=0) if gen_token_mask_row_origin is not None else gen_token_mask_row[index].unsqueeze(0)
                            target_ids_origin[counter] = all_target_ids[index]
                            app_rule_idx_row_origin[counter] = app_rule_idx_row[index]
                            app_rule_mask_row_origin[counter] = app_rule_mask_row[index]
                            token_row_origin[counter] = token_row[index]
                            gen_token_mask_row_origin[counter] = gen_token_mask_row[index]
                        counter += 1

                all_source_ids = labeled_source_ids
                all_target_ids = labeled_target_ids
                app_rule_idx_row = labeled_app_rule_idx_row
                app_rule_mask_row = labeled_app_rule_mask_row
                token_row = labeled_token_row
                gen_token_mask_row = labeled_gen_token_mask_row

            data = [all_source_ids, all_target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row]

            if distill:
                if args.task == 'concode':
                    match_index_filename =  "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_match_index.bin"
                elif args.task == 'translate' and args.sub_task == 'cs-java':
                    match_index_filename =  "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_translate_cs-java_java_match_index.bin"
                match_index = pickle.load(open(match_index_filename, 'rb'))
                match_index_tensor = torch.tensor(match_index, dtype=torch.long)
                if multitask and decode_label_path is None:
                    match_index_tensor_origin = torch.cat([match_index_tensor.clone(), match_index_tensor.clone()], dim=0)
                    match_index_tensor = torch.cat([match_index_tensor, match_index_tensor], dim=0)
                    for index in range(match_index_tensor.shape[0]):
                        # match_index_tensor[index]中每个元素自增1
                        match_index_tensor[index] = match_index_tensor[index] + 1
                        # 如果超过了最大长度，则置为最大长度
                        match_index_tensor[index][match_index_tensor[index] > args.max_target_length] = args.max_target_length
                elif multitask and decode_label_path is not None:
                    match_index_tensor_origin = torch.empty((dataset_size, ) + match_index_tensor.shape[1:], dtype=torch.long)
                    labeled_match_index_tensor = torch.empty((dataset_size, ) + match_index_tensor.shape[1:], dtype=torch.long)
                    counter = 0
                    for index, value in enumerate(tqdm(decode_label)):
                        match_index_tensor_i = match_index_tensor[index] + 1
                        match_index_tensor_i[match_index_tensor_i > args.max_target_length] = args.max_target_length
                        if value == 0 or value == 2:
                            # labeled_match_index_tensor = torch.cat([labeled_match_index_tensor, match_index_tensor_i.unsqueeze(0)], dim=0) if labeled_match_index_tensor is not None else match_index_tensor_i.unsqueeze(0)
                            # match_index_tensor_origin = torch.cat([match_index_tensor_origin, match_index_tensor[index].unsqueeze(0)], dim=0) if match_index_tensor_origin is not None else match_index_tensor[index].unsqueeze(0)
                            labeled_match_index_tensor[counter] = match_index_tensor_i
                            match_index_tensor_origin[counter] = match_index_tensor[index]
                            counter += 1
                        if value == 1 or value == 2:
                            # labeled_match_index_tensor = torch.cat([labeled_match_index_tensor, match_index_tensor_i.unsqueeze(0)], dim=0) if labeled_match_index_tensor is not None else match_index_tensor_i.unsqueeze(0)
                            # match_index_tensor_origin = torch.cat([match_index_tensor_origin, match_index_tensor[index].unsqueeze(0)], dim=0) if match_index_tensor_origin is not None else match_index_tensor[index].unsqueeze(0)
                            labeled_match_index_tensor[counter] = match_index_tensor_i
                            match_index_tensor_origin[counter] = match_index_tensor[index]
                            counter += 1

                    match_index_tensor = labeled_match_index_tensor
                    
                if not multitask:
                    data.append(match_index_tensor)
                    print(match_index_tensor.shape)
                else:
                    data.append(match_index_tensor)
                    data.append(match_index_tensor_origin)
                    print(match_index_tensor.shape)
                    print(match_index_tensor_origin.shape)

            if distill and multitask:
                data.append(target_ids_origin)
                data.append(app_rule_idx_row_origin)
                data.append(app_rule_mask_row_origin)
                data.append(token_row_origin)
                data.append(gen_token_mask_row_origin)
            
            if multitask and decode_label_path is not None:
                first_token_loss_mask = torch.tensor(first_token_loss_mask, dtype=torch.bool)
                data.append(first_token_loss_mask)

            print('******************************')
            for item in data:
                print(item.shape)
            data = TensorDataset(*data)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_gen_bi_tree_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False, sample_number=0, distill=False, multitask=False, decode_label_path=None):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    if args.debug:
        data_tag = '_debug'
    cache_fn = '{}/{}_srcAST.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    if decode_label_path is not None and split_tag != 'test': 
        # decode_label_path = /home/sata/sly/CG/bin/multitask_train_1_decode_label.pkl, cache_fn = .../train_all.pt->.../multitask_train_all.pt
        # decode_label_path = /home/sata/sly/CG/bin/multitask_distill_cross_train_1_decode_label.pkl, cache_fn = .../train_all.pt->.../multitask_distill_cross_train_all.pt
        mode = '_'.join(decode_label_path.split('/')[-1].split('1')[0].split('_')[:-2])
        cache_fn = '/'.join(cache_fn.split('/')[:-1]) + '/' + mode + '_'+ cache_fn.split('/')[-1]
    print(cache_fn, filename)

    if args.debug:
        filename = '/'.join(filename.split('/')[:-1]) + '/debug_' + filename.split('/')[-1]
    examples = read_examples(filename, args.data_num, args.task)

    is_sample = True
    if is_sample:
        # 取前20个
        examples = examples[:20]
        # examples = random.sample(examples, min(5000 if not sample_number else sample_number, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if args.sub_task == 'cs-java':
            source_tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_translate_" + args.sub_task + "_cs.bin"
            print('source_tree_filename: ', source_tree_filename)
        source_tree_labels = pickle.load(open(source_tree_filename, 'rb'))
        print(len(source_tree_labels))
        app_rule_idx_row_s = torch.tensor([f[0][:args.max_target_length] + [0] * (args.max_target_length - len(f[0][:args.max_target_length]))
                                                        for f in source_tree_labels], dtype=torch.long)
        app_rule_mask_row_s = torch.tensor([f[1][:args.max_target_length] + [0] * (args.max_target_length - len(f[1][:args.max_target_length]))
                                                        for f in source_tree_labels], dtype=torch.long)
        token_row_s = torch.tensor([f[2][:args.max_target_length] + [0] * (args.max_target_length - len(f[2][:args.max_target_length]))
                                                        for f in source_tree_labels], dtype=torch.long)
        gen_token_mask_row_s = torch.tensor([f[3][:args.max_target_length] + [0] * (args.max_target_length - len(f[3][:args.max_target_length]))
                                                          for f in source_tree_labels], dtype=torch.long)
        all_source_ids = [app_rule_idx_row_s, app_rule_mask_row_s, token_row_s, gen_token_mask_row_s]
        # if split_tag == 'test' or only_src:
        #     data = TensorDataset(all_source_ids)
        if split_tag == 'test' or only_src:
            all_source_ids = [e[:20] for e in all_source_ids]
            data = TensorDataset(*all_source_ids)
        else:
            if args.task == 'concode':
                tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + ".bin"
            elif args.task == 'translate' and args.sub_task == 'cs-java':
                tree_filename = "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_translate_cs-java_java.bin"
            print('tree_filename: ', tree_filename)
            tree_labels = pickle.load(open(tree_filename, 'rb'))

            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            # app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row
            app_rule_idx_row = torch.tensor([f[0][:args.max_target_length] + [0] * (args.max_target_length - len(f[0][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            app_rule_mask_row = torch.tensor([f[1][:args.max_target_length] + [0] * (args.max_target_length - len(f[1][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            token_row = torch.tensor([f[2][:args.max_target_length] + [0] * (args.max_target_length - len(f[2][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            gen_token_mask_row = torch.tensor([f[3][:args.max_target_length] + [0] * (args.max_target_length - len(f[3][:args.max_target_length]))
                                                          for f in tree_labels], dtype=torch.long)
            if multitask and decode_label_path is None:
                # 整体右移一位，并将第一个token替换为<seq>
                all_target_ids_seq = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]] * all_target_ids.shape[0], dtype=torch.long), all_target_ids[:, :-1]], dim=1)
                app_rule_idx_row_seq = torch.cat([torch.tensor([[0]] * app_rule_idx_row.shape[0], dtype=torch.long), app_rule_idx_row[:, :-1]], dim=1)
                app_rule_mask_row_seq = torch.cat([torch.tensor([[0]] * app_rule_mask_row.shape[0], dtype=torch.long), app_rule_mask_row[:, :-1]], dim=1)
                token_row_seq = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]] * token_row.shape[0], dtype=torch.long), token_row[:, :-1]], dim=1)
                gen_token_mask_row_seq = torch.cat([torch.tensor([[1]] * gen_token_mask_row.shape[0], dtype=torch.long), gen_token_mask_row[:, :-1]], dim=1)
                # 整体右移一位，并将第一个token替换为<tree>
                all_target_ids_tree = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]] * all_target_ids.shape[0], dtype=torch.long), all_target_ids[:, :-1]], dim=1)
                app_rule_idx_row_tree = torch.cat([torch.tensor([[0]] * app_rule_idx_row.shape[0], dtype=torch.long), app_rule_idx_row[:, :-1]], dim=1)
                app_rule_mask_row_tree = torch.cat([torch.tensor([[0]] * app_rule_mask_row.shape[0], dtype=torch.long), app_rule_mask_row[:, :-1]], dim=1)
                token_row_tree = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]] * token_row.shape[0], dtype=torch.long), token_row[:, :-1]], dim=1)
                gen_token_mask_row_tree = torch.cat([torch.tensor([[1]] * gen_token_mask_row.shape[0], dtype=torch.long), gen_token_mask_row[:, :-1]], dim=1)
                if distill:
                    # clone原来的target_ids、app_rule_idx_row、app_rule_mask_row、token_row、gen_token_mask_row
                    target_ids_origin = torch.cat([all_target_ids.clone(), all_target_ids.clone()], dim=0)
                    app_rule_idx_row_origin = torch.cat([app_rule_idx_row.clone(), app_rule_idx_row.clone()], dim=0)
                    app_rule_mask_row_origin = torch.cat([app_rule_mask_row.clone(), app_rule_mask_row.clone()], dim=0)
                    token_row_origin = torch.cat([token_row.clone(), token_row.clone()], dim=0)
                    gen_token_mask_row_origin = torch.cat([gen_token_mask_row.clone(), gen_token_mask_row.clone()], dim=0)
                # 更新原数据
                all_source_ids = torch.cat([all_source_ids, all_source_ids], dim=0)
                all_target_ids = torch.cat([all_target_ids_seq, all_target_ids_tree], dim=0)
                app_rule_idx_row = torch.cat([app_rule_idx_row_seq, app_rule_idx_row_tree], dim=0)
                app_rule_mask_row = torch.cat([app_rule_mask_row_seq, app_rule_mask_row_tree], dim=0)
                token_row = torch.cat([token_row_seq, token_row_tree], dim=0)
                gen_token_mask_row = torch.cat([gen_token_mask_row_seq, gen_token_mask_row_tree], dim=0)
            elif multitask and decode_label_path is not None:
                # load decode_label
                decode_label = pickle.load(open(decode_label_path, 'rb'))
                if args.debug:
                    decode_label = decode_label[:100]
                # 根据decode_label提前确定训练集样本数量
                dataset_size = 0
                for value in decode_label:
                    if value == 0 or value == 1:
                        dataset_size += 1
                    else:
                        dataset_size += 2
                logger.info('dataset_size: {}'.format(dataset_size))
                labeled_source_ids = torch.empty(((dataset_size,) + all_source_ids.shape[1:]), dtype=torch.long)
                labeled_target_ids = torch.empty(((dataset_size,) + all_target_ids.shape[1:]), dtype=torch.long)
                labeled_app_rule_idx_row = torch.empty(((dataset_size,) + app_rule_idx_row.shape[1:]), dtype=torch.long)
                labeled_app_rule_mask_row = torch.empty(((dataset_size,) + app_rule_mask_row.shape[1:]), dtype=torch.long)
                labeled_token_row = torch.empty(((dataset_size,) + token_row.shape[1:]), dtype=torch.long)
                labeled_gen_token_mask_row = torch.empty(((dataset_size,) + gen_token_mask_row.shape[1:]), dtype=torch.long)
                first_token_loss_mask = []
                # 遍历decode_label的值，0-<seq>, 1-<tree>, 2-<seq>&<tree>
                target_ids_origin = torch.empty(((dataset_size,) + all_target_ids.shape[1:]), dtype=torch.long)
                app_rule_idx_row_origin = torch.empty(((dataset_size,) + app_rule_idx_row.shape[1:]), dtype=torch.long)
                app_rule_mask_row_origin = torch.empty(((dataset_size,) + app_rule_mask_row.shape[1:]), dtype=torch.long)
                token_row_origin = torch.empty(((dataset_size,) + token_row.shape[1:]), dtype=torch.long)
                gen_token_mask_row_origin = torch.empty(((dataset_size,) + gen_token_mask_row.shape[1:]), dtype=torch.long)
                counter = 0
                for index, value in enumerate(tqdm(decode_label)):
                    if value == 0 or value == 2:
                        if value == 2:
                            # 不计算第一个token的loss, False
                            first_token_loss_mask.append(False)
                        else:
                            first_token_loss_mask.append(True)
                        # 右移一位，并将第一个token替换为<seq>
                        # labeled_source_ids = torch.cat([labeled_source_ids, all_source_ids[index].unsqueeze(0)], dim=0) if labeled_source_ids is not None else all_source_ids[index].unsqueeze(0)
                        labeled_source_ids[counter] = all_source_ids[index]
                        
                        index_target_ids = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]], dtype=torch.long), all_target_ids[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_target_ids = torch.cat([labeled_target_ids, index_target_ids], dim=0) if labeled_target_ids is not None else index_target_ids
                        labeled_target_ids[counter] = index_target_ids
                        
                        index_app_rule_idx_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_idx_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_idx_row = torch.cat([labeled_app_rule_idx_row, index_app_rule_idx_row], dim=0) if labeled_app_rule_idx_row is not None else index_app_rule_idx_row
                        labeled_app_rule_idx_row[counter] = index_app_rule_idx_row
                        
                        index_app_rule_mask_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_mask_row = torch.cat([labeled_app_rule_mask_row, index_app_rule_mask_row], dim=0) if labeled_app_rule_mask_row is not None else index_app_rule_mask_row
                        labeled_app_rule_mask_row[counter] = index_app_rule_mask_row
                        
                        index_token_row = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<seq>')]], dtype=torch.long), token_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_token_row = torch.cat([labeled_token_row, index_token_row], dim=0) if labeled_token_row is not None else index_token_row
                        labeled_token_row[counter] = index_token_row

                        index_gen_token_mask_row = torch.cat([torch.tensor([[1]], dtype=torch.long), gen_token_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_gen_token_mask_row = torch.cat([labeled_gen_token_mask_row, index_gen_token_mask_row], dim=0) if labeled_gen_token_mask_row is not None else index_gen_token_mask_row
                        labeled_gen_token_mask_row[counter] = index_gen_token_mask_row
                        if distill:
                            # target_ids_origin = torch.cat([target_ids_origin, all_target_ids[index].unsqueeze(0)], dim=0) if target_ids_origin is not None else all_target_ids[index].unsqueeze(0)
                            target_ids_origin[counter] = all_target_ids[index]
                            # app_rule_idx_row_origin = torch.cat([app_rule_idx_row_origin, app_rule_idx_row[index].unsqueeze(0)], dim=0) if app_rule_idx_row_origin is not None else app_rule_idx_row[index].unsqueeze(0)
                            app_rule_idx_row_origin[counter] = app_rule_idx_row[index]
                            # app_rule_mask_row_origin = torch.cat([app_rule_mask_row_origin, app_rule_mask_row[index].unsqueeze(0)], dim=0) if app_rule_mask_row_origin is not None else app_rule_mask_row[index].unsqueeze(0)
                            app_rule_mask_row_origin[counter] = app_rule_mask_row[index]
                            # token_row_origin = torch.cat([token_row_origin, token_row[index].unsqueeze(0)], dim=0) if token_row_origin is not None else token_row[index].unsqueeze(0)
                            token_row_origin[counter] = token_row[index]
                            # gen_token_mask_row_origin = torch.cat([gen_token_mask_row_origin, gen_token_mask_row[index].unsqueeze(0)], dim=0) if gen_token_mask_row_origin is not None else gen_token_mask_row[index].unsqueeze(0)
                            gen_token_mask_row_origin[counter] = gen_token_mask_row[index]
                        counter += 1
                    if value == 1 or value == 2:
                        if value == 2:
                            # 不计算第一个token的loss, False
                            first_token_loss_mask.append(False)
                        else:
                            first_token_loss_mask.append(True)
                        # 右移一位，并将第一个token替换为<tree>
                        # labeled_source_ids = torch.cat([labeled_source_ids, all_source_ids[index].unsqueeze(0)], dim=0) if labeled_source_ids is not None else all_source_ids[index].unsqueeze(0)
                        labeled_source_ids[counter] = all_source_ids[index]

                        index_target_ids = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]], dtype=torch.long), all_target_ids[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_target_ids = torch.cat([labeled_target_ids, index_target_ids], dim=0) if labeled_target_ids is not None else index_target_ids
                        labeled_target_ids[counter] = index_target_ids

                        index_app_rule_idx_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_idx_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_idx_row = torch.cat([labeled_app_rule_idx_row, index_app_rule_idx_row], dim=0) if labeled_app_rule_idx_row is not None else index_app_rule_idx_row
                        labeled_app_rule_idx_row[counter] = index_app_rule_idx_row

                        index_app_rule_mask_row = torch.cat([torch.tensor([[0]], dtype=torch.long), app_rule_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_app_rule_mask_row = torch.cat([labeled_app_rule_mask_row, index_app_rule_mask_row], dim=0) if labeled_app_rule_mask_row is not None else index_app_rule_mask_row
                        labeled_app_rule_mask_row[counter] = index_app_rule_mask_row

                        index_token_row = torch.cat([torch.tensor([[tokenizer.convert_tokens_to_ids('<tree>')]], dtype=torch.long), token_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_token_row = torch.cat([labeled_token_row, index_token_row], dim=0) if labeled_token_row is not None else index_token_row
                        labeled_token_row[counter] = index_token_row

                        index_gen_token_mask_row = torch.cat([torch.tensor([[1]], dtype=torch.long), gen_token_mask_row[index][:-1].unsqueeze(0)], dim=1)
                        # labeled_gen_token_mask_row = torch.cat([labeled_gen_token_mask_row, index_gen_token_mask_row], dim=0) if labeled_gen_token_mask_row is not None else index_gen_token_mask_row
                        labeled_gen_token_mask_row[counter] = index_gen_token_mask_row
                        
                        if distill:
                            target_ids_origin[counter] = all_target_ids[index]
                            app_rule_idx_row_origin[counter] = app_rule_idx_row[index]
                            app_rule_mask_row_origin[counter] = app_rule_mask_row[index]
                            token_row_origin[counter] = token_row[index]
                            gen_token_mask_row_origin[counter] = gen_token_mask_row[index]
                        counter += 1

                all_source_ids = labeled_source_ids
                all_target_ids = labeled_target_ids
                app_rule_idx_row = labeled_app_rule_idx_row
                app_rule_mask_row = labeled_app_rule_mask_row
                token_row = labeled_token_row
                gen_token_mask_row = labeled_gen_token_mask_row

            data = all_source_ids + [all_target_ids, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row]

            if distill:
                match_index_filename =  "/home/sly/CG/CodeT5/asdl/lang/java/bin/" + split_tag + ('_debug' if args.debug else '') + "_match_index.bin"
                match_index = pickle.load(open(match_index_filename, 'rb'))
                match_index_tensor = torch.tensor(match_index, dtype=torch.long)
                if multitask and decode_label_path is None:
                    match_index_tensor_origin = torch.cat([match_index_tensor.clone(), match_index_tensor.clone()], dim=0)
                    match_index_tensor = torch.cat([match_index_tensor, match_index_tensor], dim=0)
                    for index in range(match_index_tensor.shape[0]):
                        # match_index_tensor[index]中每个元素自增1
                        match_index_tensor[index] = match_index_tensor[index] + 1
                        # 如果超过了最大长度，则置为最大长度
                        match_index_tensor[index][match_index_tensor[index] > args.max_target_length] = args.max_target_length
                elif multitask and decode_label_path is not None:
                    match_index_tensor_origin = torch.empty((dataset_size, ) + match_index_tensor.shape[1:], dtype=torch.long)
                    labeled_match_index_tensor = torch.empty((dataset_size, ) + match_index_tensor.shape[1:], dtype=torch.long)
                    counter = 0
                    for index, value in enumerate(tqdm(decode_label)):
                        match_index_tensor_i = match_index_tensor[index] + 1
                        match_index_tensor_i[match_index_tensor_i > args.max_target_length] = args.max_target_length
                        if value == 0 or value == 2:
                            # labeled_match_index_tensor = torch.cat([labeled_match_index_tensor, match_index_tensor_i.unsqueeze(0)], dim=0) if labeled_match_index_tensor is not None else match_index_tensor_i.unsqueeze(0)
                            # match_index_tensor_origin = torch.cat([match_index_tensor_origin, match_index_tensor[index].unsqueeze(0)], dim=0) if match_index_tensor_origin is not None else match_index_tensor[index].unsqueeze(0)
                            labeled_match_index_tensor[counter] = match_index_tensor_i
                            match_index_tensor_origin[counter] = match_index_tensor[index]
                            counter += 1
                        if value == 1 or value == 2:
                            # labeled_match_index_tensor = torch.cat([labeled_match_index_tensor, match_index_tensor_i.unsqueeze(0)], dim=0) if labeled_match_index_tensor is not None else match_index_tensor_i.unsqueeze(0)
                            # match_index_tensor_origin = torch.cat([match_index_tensor_origin, match_index_tensor[index].unsqueeze(0)], dim=0) if match_index_tensor_origin is not None else match_index_tensor[index].unsqueeze(0)
                            labeled_match_index_tensor[counter] = match_index_tensor_i
                            match_index_tensor_origin[counter] = match_index_tensor[index]
                            counter += 1

                    match_index_tensor = labeled_match_index_tensor
                    
                if not multitask:
                    data.append(match_index_tensor_origin)
                    print(match_index_tensor_origin.shape)
                else:
                    data.append(match_index_tensor)
                    data.append(match_index_tensor_origin)
                    print(match_index_tensor.shape)
                    print(match_index_tensor_origin.shape)

            if distill and multitask:
                data.append(target_ids_origin)
                data.append(app_rule_idx_row_origin)
                data.append(app_rule_mask_row_origin)
                data.append(token_row_origin)
                data.append(gen_token_mask_row_origin)
            
            if multitask and decode_label_path is not None:
                first_token_loss_mask = torch.tensor(first_token_loss_mask, dtype=torch.bool)
                data.append(first_token_loss_mask)

            print('******************************')
            for item in data:
                print(item.shape)
            data = TensorDataset(*data)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_gen_tune_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False, sample_number=0, distill=False, multitask=False, decode_label_path=None):
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    if args.debug:
        data_tag = '_debug'
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    if decode_label_path is not None and split_tag != 'test': 
        mode = '_'.join(decode_label_path.split('/')[-1].split('1')[0].split('_')[:-2])
        cache_fn = '/'.join(cache_fn.split('/')[:-1]) + '/' + mode + '_'+ cache_fn.split('/')[-1]
    print(cache_fn, filename)

    if args.debug:
        filename = '/'.join(filename.split('/')[:-1]) + '/debug_' + filename.split('/')[-1]
    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(5000 if not sample_number else sample_number, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    # if os.path.exists(cache_fn) and not is_sample:
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:
    if is_sample:
        logger.info("Sample data for computing bleu from %s", filename)
    else:
        logger.info("Create cache data into %s", cache_fn)
    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    if split_tag == 'test' or only_src:
        if split_tag == 'train':
            seq_codes = [line.strip() for line in open(args.train_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.train_tree_output, 'r').readlines()]
        elif split_tag == 'dev':
            seq_codes = [line.strip() for line in open(args.dev_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.dev_tree_output, 'r').readlines()]
        elif split_tag == 'test':
            seq_codes = [line.strip() for line in open(args.test_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.test_tree_output, 'r').readlines()]
        tuple_targets = [(seq_code, tree_code, tokenizer, args) for seq_code, tree_code in zip(seq_codes, tree_codes)]
        target_ids = pool.map(convert_target_to_ids, tqdm(tuple_targets, total=len(tuple_targets)))
        seq_code_ids = torch.tensor([t['seq_target_ids'] for t in target_ids], dtype=torch.long)
        tree_code_ids = torch.tensor([t['tree_target_ids'] for t in target_ids], dtype=torch.long)
        
        data = TensorDataset(*[all_source_ids, seq_code_ids, tree_code_ids])
    else:
        if split_tag == 'train':
            seq_codes = [line.strip() for line in open(args.train_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.train_tree_output, 'r').readlines()]
        elif split_tag == 'dev':
            seq_codes = [line.strip() for line in open(args.dev_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.dev_tree_output, 'r').readlines()]
        elif split_tag == 'test':
            seq_codes = [line.strip() for line in open(args.test_seq_output, 'r').readlines()]
            tree_codes = [line.strip() for line in open(args.test_tree_output, 'r').readlines()]
        
        example_weight = []
        for example, seq_code, tree_code in zip(examples, seq_codes, tree_codes):
            target_code = example.target
            seq_bleu, _, _, _, _, _   = compute_bleu([target_code.strip().split()], [seq_code.strip().split()], 4, True)
            tree_bleu, _, _, _, _, _  = compute_bleu([target_code.strip().split()], [tree_code.strip().split()], 4, True)
            example_weight.append(round(100 * abs(seq_bleu - tree_bleu), 2))
        example_weight = torch.tensor(example_weight, dtype=torch.float)

        tuple_targets = [(seq_code, tree_code, tokenizer, args) for seq_code, tree_code in zip(seq_codes, tree_codes)]
        output_ids = pool.map(convert_target_to_ids, tqdm(tuple_targets, total=len(tuple_targets)))
        seq_code_ids = torch.tensor([t['seq_target_ids'] for t in output_ids], dtype=torch.long)
        tree_code_ids = torch.tensor([t['tree_target_ids'] for t in output_ids], dtype=torch.long)

        decode_label = pickle.load(open(decode_label_path, 'rb'))
        if args.debug:
            decode_label = decode_label[:100]
        count_res = Counter(decode_label)
        logger.info("decode_label: %s", count_res)
        class_ratio = count_res[0] / count_res[1]
        logger.info("class_ratio: %s", class_ratio)

        # filter by decode_label = 2
        decode_label_tensors = torch.tensor(decode_label, dtype=torch.long)
        choose_mask = decode_label_tensors!=2
        if args.sample_file is not None and split_tag == 'train':
            sample_data = np.loadtxt(args.sample_file, dtype=np.int64)
            sample_data = torch.from_numpy(sample_data)
            consist_mask = decode_label_tensors==sample_data
            choose_mask = choose_mask & consist_mask
        all_source_ids = all_source_ids[choose_mask]
        seq_code_ids = seq_code_ids[choose_mask]
        tree_code_ids = tree_code_ids[choose_mask]
        decode_label_tensors = decode_label_tensors[choose_mask]
        example_weight = example_weight[choose_mask]

        data = [all_source_ids, seq_code_ids, tree_code_ids, decode_label_tensors, example_weight]
        if args.over_sample and split_tag == 'train':
            index_tensor = torch.tensor([i for i in range(len(decode_label_tensors))], dtype=torch.long)
            over_sampler = RandomOverSampler()
            # X_train = [all_source_ids, seq_code_ids, tree_code_ids, all_target_ids]
            # source_len, seq_len, tree_len, tgt_len = all_source_ids.shape[-1], seq_code_ids.shape[-1], tree_code_ids.shape[-1], all_target_ids.shape[-1]
            # X_train = torch.cat(X_train, dim=-1).numpy().reshape(-1, source_len+seq_len+tree_len+tgt_len)
            index_resampled, decode_label_tensors_resampled = over_sampler.fit_resample(index_tensor.unsqueeze(-1), decode_label_tensors)
            index_tensor_resampled = index_resampled.squeeze(-1)
            all_source_ids_resampled = all_source_ids[index_tensor_resampled]
            seq_code_ids_resampled = seq_code_ids[index_tensor_resampled]
            tree_code_ids_resampled = tree_code_ids[index_tensor_resampled]
            example_weight_resampled = example_weight[index_tensor_resampled]

            print("number after oversampling:", len(decode_label_tensors_resampled))

            # data = [torch.from_numpy(all_source_ids_resampled), torch.from_numpy(seq_code_ids_resampled), torch.from_numpy(tree_code_ids_resampled), torch.from_numpy(decode_label_tensors_resampled), torch.from_numpy(target_ids_resampled)]
            data = [all_source_ids_resampled, seq_code_ids_resampled, tree_code_ids_resampled, torch.from_numpy(decode_label_tensors_resampled), example_weight_resampled]

            print('******************************')
            for item in data:
                print(item.shape)
        elif args.down_sample and split_tag == 'train':
            down_sampler = RandomUnderSampler(random_state=42)
            all_source_ids_resampled, decode_label_tensors_resampled = down_sampler.fit_resample(all_source_ids, decode_label_tensors)
            print("number after downsampling:", len(all_source_ids_resampled))

            data = [torch.from_numpy(all_source_ids_resampled), torch.from_numpy(decode_label_tensors_resampled)]

            print('******************************')
            for item in data:
                print(item.shape)

        data = TensorDataset(*data)

    if split_tag == 'train':
        return examples, data, class_ratio
    else:
        return examples, data


def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_multi_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = ['summarize', 'translate', 'refine', 'concode', 'defect']
        for task in task_list:
            if task == 'summarize':
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
            elif task == 'translate':
                sub_tasks = ['java-cs', 'cs-java']
            elif task == 'refine':
                sub_tasks = ['small', 'medium']
            else:
                sub_tasks = ['none']
            args.task = task
            for sub_task in sub_tasks:
                args.sub_task = sub_task
                if task == 'summarize':
                    args.max_source_length = 256
                    args.max_target_length = 128
                elif task == 'translate':
                    args.max_source_length = 320
                    args.max_target_length = 256
                elif task == 'refine':
                    if sub_task == 'small':
                        args.max_source_length = 130
                        args.max_target_length = 120
                    else:
                        args.max_source_length = 240
                        args.max_target_length = 240
                elif task == 'concode':
                    args.max_source_length = 320
                    args.max_target_length = 150
                elif task == 'defect':
                    args.max_source_length = 512
                    args.max_target_length = 3  # as do not need to add lang ids

                filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
                examples = read_examples(filename, args.data_num, args.task)
                if is_sample:
                    examples = random.sample(examples, min(5000, len(examples)))
                if split_tag == 'train':
                    calc_stats(examples, tokenizer, is_tokenize=True)
                else:
                    calc_stats(examples)

                tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
                if args.data_num == -1:
                    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                else:
                    features = [convert_examples_to_features(x) for x in tuple_examples]
                all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                if only_src:
                    data = TensorDataset(all_source_ids)
                else:
                    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                    data = TensorDataset(all_source_ids, all_target_ids)
                examples_data_dict['{}_{}'.format(task, sub_task) if sub_task != 'none' else task] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'conala':
        # data_dir = '{}/{}/conala-corpus'.format(data_root, task)
        # train_fn = '{}/conala-train.jsonl'.format(data_dir)
        # # train_fn = '{}/conala-mined.jsonl'.format(data_dir)
        # dev_fn = '{}/conala-dev.jsonl'.format(data_dir)
        # test_fn = '{}/conala-test.jsonl'.format(data_dir)
        # conala-corpus/doc_prompt/docprompting_data/conala/fid.cmd_train.codet5.t10.json
        data_dir = '{}/{}/conala-corpus/doc_prompt/docprompting_data/conala'.format(data_root, task)
        train_fn = '{}/fid.cmd_train.codet5.t10.json'.format(data_dir)
        dev_fn = '{}/fid.cmd_dev.codet5.t10.json'.format(data_dir)
        test_fn = '{}/fid.cmd_test.codet5.t10.json'.format(data_dir)

    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)