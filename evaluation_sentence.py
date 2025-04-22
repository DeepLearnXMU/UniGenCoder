from evaluator.bleu import _bleu_sentence, _bleu
import argparse
import csv
import numpy as np
import os
import pickle
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
from asdl.lang.java.java_transition_system import tree_sitter_ast_statistics

def parse_args():
    parser = argparse.ArgumentParser(description="Create vocabulary")
    parser.add_argument("--gold_tree", type=str, required=True,)
    parser.add_argument("--gold_seq", type=str, required=True,)
    parser.add_argument("--pred_tree", type=str, required=True,)
    parser.add_argument("--pred_seq", type=str, required=True,)
    parser.add_argument('--output', type=str, default='bleu_sorted.csv')

    return parser.parse_args()


def main(args):
    ''' dev 10 组：
        /home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize10_dev/test_best-bleu.output
        /home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize10_dev/test_best-bleu.output
        dev 1 组：
        /home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_dev/test_best-bleu.output
        /home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_dev/test_best-bleu.output
        test 1 组：
        /home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_test/test_best-bleu.output
        /home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_test/test_best-bleu.output
        train 1 组：
        /home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_train/test_best-bleu.output
        /home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_train/test_best-bleu.output
    '''
    seq_file = [
        # # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_test/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize10_dev/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_dev/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_train/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318_1234/prediction/test_average.bin_seq.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize1_train/test_best-ppl_seq.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize1_train/test_best-ppl_seq.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize10_dev/test_best-ppl_seq.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize10_dev/test_best-ppl_seq.output',
        '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313_1234/prediction/test_average.bin_seq.output',
    ]
    tree_file = [
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_test/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize10_dev/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_dev/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_train/test_best-bleu.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318_1234/prediction/test_average.bin_tree.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize1_train/test_best-ppl_tree.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize1_train/test_best-ppl_tree.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize10_dev/test_best-ppl_tree.output',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize10_dev/test_best-ppl_tree.output',
        '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313_1234/prediction/test_average.bin_tree.output',
    ]
    gold_file = [
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_test/test_best-bleu.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction/test_best-bleu.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_dev/test_best-bleu.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction_beamsize1_dev/test_best-bleu.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_tree/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/prediction_beamsize1_train/test_best-bleu.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318_1234/prediction/test_average.bin_seq.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize1_train/test_best-ppl_tree.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize1_train/test_best-ppl_seq.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230412223318/prediction_beamsize10_dev/test_best-ppl_tree.gold',
        # '/home/sly/CG/CodeT5/saved_models/codeT5_multitask/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313/prediction_beamsize10_dev/test_best-ppl_tree.gold',
        '/home/sly/CG/CodeT5/saved_models/codeT5_multitask_distill/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230420214313_1234/prediction/test_average.bin_tree.gold',
    ]
    labels = [
        # 'test_1',
        # 'test_10',
        # 'dev_10', 
        # 'dev_1',
        # 'train_1',
        # 'test_average_label',
        # 'multitask_train_1',
        # 'multitask_distill_cross_train_1',
        # 'multitask_dev_10',
        # 'multitask_distill_cross_dev_10',
        'multitask_distill_cross_test_10'
    ]
    for seq_fn, tree_fn, label_fn, gold_fn in zip(seq_file, tree_file, labels, gold_file):
        bleu_records_tree = _bleu_sentence(gold_fn, tree_fn)
        bleu_records_seq = _bleu_sentence(gold_fn, seq_fn)
        # bleu_records = sorted(zip(bleu_records_seq, bleu_records_tree), key=lambda x: x[0][0] - x[1][0])
        bleu_records = zip(bleu_records_seq, bleu_records_tree)
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        JAVA_LANGUAGE = Language('asdl/lang/java/build/my-languages.so', 'java')
        parser = Parser()
        parser.set_language(JAVA_LANGUAGE)
        score_diff = [x[0][0] - x[1][0] for x in bleu_records]
        bins = np.arange(-1.0, 1.1, 0.1)
        counts, edges = np.histogram(score_diff, bins)
        print(counts)
        print(edges)
        for num in counts:
            print(num)
        optim_list = []
        dataset_number = 0
        with open(args.output + label_fn + '.csv', 'w') as f:
            decode_label_list = [] # 0 for <seq>; 1 for <tree>; 2 for <seq>&<tree>
            writer = csv.writer(f)
            writer.writerow(['seq_bleu_score', 'seq_pred', 'tree_bleu_score', 'tree_pred', 'reference', 'score_diff', 'ref_length', 'seq_length', 'tree_length', 'tree_height', 'non_leaf_num', 'leaf_num'])
            for (seq_bleu_score, seq_reference, seq_pred), \
                (tree_bleu_score, tree_reference, tree_pred) in zip(bleu_records_seq, bleu_records_tree):
                if seq_reference != tree_reference:
                    raise ValueError('Reference does not match!')
                bleu_diff = seq_bleu_score - tree_bleu_score
                if bleu_diff > 0:
                    decode_label_list.append(0)
                    optim_list.append(seq_pred)
                    dataset_number += 1
                elif bleu_diff < 0:
                    decode_label_list.append(1)
                    optim_list.append(tree_pred)
                    dataset_number += 1
                else:
                    decode_label_list.append(2)
                    optim_list.append(seq_pred)
                    dataset_number += 2
                length = len(tokenizer.encode(seq_reference))
                length_seq = len(tokenizer.encode(seq_pred))
                length_tree = len(tokenizer.encode(tree_pred))
                tree = parser.parse(bytes(seq_reference, "utf8"))
                cursor = tree.walk()
                non_leaf_num, leaf_num, height = tree_sitter_ast_statistics(cursor.node)
                writer.writerow([seq_bleu_score, seq_pred, tree_bleu_score, tree_pred, seq_reference, seq_bleu_score - tree_bleu_score, length, length_seq, length_tree, height, non_leaf_num, leaf_num])
            f.close()
            pickle.dump(decode_label_list, open('/home/sata/sly/CG/bin/' + label_fn + '_decode_label.pkl', 'wb'))
        optim_fn = '/home/sly/CG/CodeT5/saved_models/codeT5/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30_20230212171720/prediction/test_best-bleu-optim.output'
        with open(optim_fn, 'w') as f:
            for item in optim_list:
                f.write(item + '\n')
            f.close()
        print('dataset_number: ', dataset_number)
        print(_bleu(gold_fn, tree_fn))
        print(_bleu(gold_fn, seq_fn))
        print(_bleu(gold_fn, optim_fn))

if __name__ == '__main__':
    main(parse_args())