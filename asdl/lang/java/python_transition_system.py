# GO_LANGUAGE = Language('build/my-languages.so', 'go')
# JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
# PY_LANGUAGE = Language('build/my-languages.so', 'python')

from tree_sitter import Language, Parser
import time
import json
import sys
from asdl.hypothesis import *
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree
import numpy as np
import math
import pickle
from tqdm import tqdm
import argparse
from components.vocab import Vocab, VocabEntry
from components.action_info import ActionInfo, get_action_infos
from transformers import RobertaTokenizer
import os


def java_ast_to_asdl_ast(java_ast_node, grammar):
    # node should be composite
    py_node_name = java_ast_node.type
    # assert py_node_name.startswith('_ast.')

    # print(py_node_name)
    # print(type(py_node_name))
    # print(grammar._productions.keys())
    type_production_list = grammar._productions[ASDLCompositeType(py_node_name)]
    production_str = py_node_name + ' -> ' + "None(" + ' '.join(['leaf_' + child.type + ' leaf_' + child.type if child.child_count==0 or child.type=='string'
                                                                 else child.type + ' ' + child.type for child in java_ast_node.children]) + ')'
    # print(production_str)
    # exit()
    production = None
    for production_i in type_production_list:
        if production_str == str(production_i):
            production = production_i
            break
        else:
            pass
    if production is None:
        print(production_str, type_production_list)
    fields = []
    for index, field in enumerate(production.fields):
        # field_value = getattr(java_ast_node, field.name)
        field_value = java_ast_node.children[index]
        asdl_field = RealizedField(field)
        if field.cardinality == 'single' or field.cardinality == 'optional':
            if field_value is not None:  # sometimes it could be 0
                if grammar.is_composite_type(field.type):
                    child_node = java_ast_to_asdl_ast(field_value, grammar)
                    asdl_field.add_value(child_node)
                else:
                    if field_value.is_named:
                        asdl_field.add_value(str(field_value.text, encoding = "utf-8"))
                    else:
                        asdl_field.add_value(str(field_value.type))
        # field with multiple cardinality
        elif field_value is not None:
            if grammar.is_composite_type(field.type):
                for val in field_value:
                    child_node = java_ast_to_asdl_ast(val, grammar)
                    asdl_field.add_value(child_node)
            else:
                for val in field_value:
                    asdl_field.add_value(str(val))

        fields.append(asdl_field)

    asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

    return asdl_node

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

# def asdl_ast_to_java_ast(asdl_ast_node, grammar):
#     py_node_type = getattr(sys.modules['tree_sitter'], asdl_ast_node.production.type.name)
#     py_node_type = asdl_ast_node.production.type.name
#     py_ast_node = py_node_type()
#
#     for field in asdl_ast_node.fields:
#         # for composite node
#         field_value = None
#         if grammar.is_composite_type(field.type):
#             if field.value and field.cardinality == 'multiple':
#                 field_value = []
#                 for val in field.value:
#                     node = asdl_ast_to_python_ast(val, grammar)
#                     field_value.append(node)
#             elif field.value and field.cardinality in ('single', 'optional'):
#                 field_value = asdl_ast_to_python_ast(field.value, grammar)
#         else:
#             # for primitive node, note that primitive field may have `None` value
#             if field.value is not None:
#                 if field.type.name == 'object':
#                     if '.' in field.value or 'e' in field.value:
#                         field_value = float(field.value)
#                     elif isint(field.value):
#                         field_value = int(field.value)
#                     else:
#                         raise ValueError('cannot convert [%s] to float or int' % field.value)
#                 elif field.type.name == 'int':
#                     field_value = int(field.value)
#                 else:
#                     field_value = field.value
#
#             # FIXME: hack! if int? is missing value in ImportFrom(identifier? module, alias* names, int? level), fill with 0
#             elif field.name == 'level':
#                 field_value = 0
#
#         # must set unused fields to default value...
#         if field_value is None and field.cardinality == 'multiple':
#             field_value = list()
#
#         setattr(py_ast_node, field.name, field_value)
#
#     return py_ast_node

class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions
        self.idx = idx
        self.meta = meta

class JavaTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        pass

    def ast_to_surface_code(self, asdl_ast):
        pass

    def surface_code_to_ast(self, code):
        pass

    def tokenize_code(self, code, mode):
        pass

    def get_primitive_field_actions(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':  # expr -> Global(identifier* names)
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            tokens = []
            # if realized_field.type.name == 'string':
            #     for field_val in field_values:
            #         tokens.extend(field_val.split(' ') + ['</primitive>'])
            # else:
            for field_val in field_values:
                tokens.append(field_val)

            for tok in tokens:
                actions.append(GenTokenAction(tok))
            if realized_field.target_value is None or len(realized_field.target_value) > 1:
                actions.append(GenTokenAction('</s>'))

        return actions

def check(grammar, cursor, out, code, file_name, index,  dict_blank=None, dict_=None, list_no_win=None, tokenizer=None):
    # try:
    # print(code)
    asdl_ast = java_ast_to_asdl_ast(cursor.node, grammar)
    javaTransitionSystem = JavaTransitionSystem(grammar)
    actions = javaTransitionSystem.get_actions(asdl_ast)
    hypothesis = Hypothesis()
    tgt_actions = []
    # print('\n'.join([str(action) for action in actions]))
    # exit()

    for t, action in enumerate(actions, 1):
        # the type of the action should belong to one of the valid continuing types
        # of the transition system
        if action.__class__ not in javaTransitionSystem.get_valid_continuation_types(hypothesis):
            print(action.__class__)
            print(javaTransitionSystem.get_valid_continuation_types(hypothesis))
        assert action.__class__ in javaTransitionSystem.get_valid_continuation_types(hypothesis)

        # if it's an ApplyRule action, the production rule should belong to the
        # set of rules with the same LHS type as the current rule
        if isinstance(action, ApplyRuleAction) and hypothesis.frontier_node:
            assert action.production in grammar[hypothesis.frontier_field.type]

        p_t = hypothesis.frontier_node.created_time if hypothesis.frontier_node else -1
        # print('t=%d, p_t=%d, Action=%s' % (t, p_t, action))
        hypothesis.apply_action(action)
        tgt_actions.append(action)
            # except:
            #     print(t)

    if dict_blank is not None and dict_ is not None:
        generate_node_V2(asdl_ast, grammar, code, dict_blank, dict_)
        return None
    #output_gammars = asdl_to_code(asdl_ast, grammar)
    #print(output_gammars)
    #print("code:", code)

    if list_no_win is not None:
        code1_list = asdl_to_code(asdl_ast, grammar, list_no_win, tokenizer=tokenizer)
        code1 = ''.join(code1_list)
        code2_list = asdl_to_code(hypothesis.tree, grammar, list_no_win, tokenizer=tokenizer)
        code2 = ''.join(code2_list)
    else:
        code1_list = asdl_to_code(asdl_ast, grammar, tokenizer=tokenizer)
        code1 = ' '.join([c for c in code1_list if c != ' '])
        code2 = ' '.join(asdl_to_code(hypothesis.tree, grammar, tokenizer=tokenizer))
    # assert code == code1
    # assert code == code2
    code = code.replace('\'', '"').replace('\n', '')
    code1_ = code1.replace('\'', '"')
    code2_ = code2.replace('\'', '"')
    if ''.join(code.split(' ')) != ''.join(code1_.split(' ')) or ''.join(code.split(' ')) != ''.join(
            code2_.split(' ')):
        if out!=None:
            out.write(file_name + ' ' + str(index))
            out.write('\n')
        print('\ncode={}, \ncode1={}\ncode2={}'.format(code, code1_, code2_))
        # return None
        return {'tgt_canonical_code': code1,
                'tree_tgt_code': ''.join(code1_list),
                'tgt_actions': tgt_actions}
    else:
        # if code1!=code or code2!=code:
        #     print(index)
        #     print(code)
        #     print(code1)
        #     print(code2)
        return {'tgt_canonical_code': code1,
                'tree_tgt_code': ''.join(code1_list),
                'tgt_actions': tgt_actions}
        # 'src_query_tokens': src_query_tokens,
        # 'tgt_ast': cursor,
    # except Exception as e:
    #     out.write(str(e))
    #     out.write(file_name + ' ' + str(index))
    #     out.write('\n')
    # return None

def extract_primitive_type(text):
    primitive_types = set()
    for line in text.split('\n'):
        if not line:
            continue
        types = line.strip().split('-->')[1].split(' ')
        for type in types:
            if type.startswith('leaf::'):
                primitive_types.add(type[type.find('leaf::') + 6:])
                if not type[type.find('leaf::') + 6:]:
                    print('wait')
    return primitive_types

def asdl_to_code(asdl_ast_node, grammar, list_no_win=None, tokenizer=None, mode='str'):
    def asdl_node_to_code(asdl_node, grammar, field=None):
        if not asdl_node:
            # print(asdl_node)
            # print('is_composite_type: {}_{}'.format(field.type, grammar.is_composite_type(field.type)))
            # print(vars(field))
            return code

        for index_f, field in enumerate(asdl_node.fields):
            if grammar.is_composite_type(field.type):
                asdl_node_to_code(field.value, grammar, field)
            else:
                if mode == 'tensor':
                    if field.value is None:
                        # print(field)
                        field.value = field.target_value if field.target_value is not None else ''
                    field.value = tokenizer.decode(field.value, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if type(field.value) == list:
                    field.value = [v.strip() for v in field.value]
                    code.extend(field.value)
                else:
                    field.value = field.value.strip()
                    # not empty
                    if field.value:
                        code.append(field.value)
                # if list_no_win is not None:
                    # if (field.type, str(asdl_node), index_f) not in list_no_win:
                if code[-1] not in '{ } ( ) = . , >= [ ] != + / - -- == > < || ! ++ <= ? : && @ += ... * |= % /= & -= >>> >> << ~ | >>= &= -> ^ *= ^= >>>= :: ;'.split(' '):
                    code.append(' ')
        return code
    code = []
    asdl_node_to_code(asdl_ast_node, grammar)
    not_none_code = []
    # print('code: ', code)
    for c in code:
        if c:
            not_none_code.append(c)
            # c_list = tokenizer.encode(c, add_special_tokens=False)
            # c_str_list = tokenizer.convert_ids_to_tokens(c_list)
            # for c_ in c_str_list:
            #     not_none_code.append(c_)
            #     if c_ not in '{ } ( ) = . , >= [ ] != + / - -- == > < || ! ++ <= ? : && @ += ... * |= % /= & -= >>> >> << ~ | >>= &= -> ^ *= ^= >>>= :: ;'.split(' '):
            #         not_none_code.append(' ')
    # print(''.join(not_none_code))
    return  not_none_code #' '.join(not_none_code)

def generate_node_V2(asdl_ast_node, grammar, o_code, dict_blank, dict_):
    def asdl_node_to_code(asdl_node, grammar, father, o_code, index):
        if not asdl_node:
            # print('asdl_node is None')
            return code

        for index_, field in enumerate(asdl_node.fields):
            if grammar.is_composite_type(field.type):
                asdl_node_to_code(field.value, grammar, asdl_node, o_code, index)
            else:
                if type(field.value) == list:
                    code.extend(field.value)
                else:
                    code.append(field.value)
                index[0] = index[0] + len(code[-1])
                if index[0] < len(o_code) :
                    if o_code[index[0]] == ' ':
                        special_grammer.append(field.type)
                        dict_blank[(field.type, str(father), index_)] = dict_blank.get((field.type, str(father), index_), 0) + 1
                        while o_code[index[0]] == ' ':
                            index[0] += 1
                    else:
                        dict_[(field.type, str(father), index_)] = dict_.get((field.type, str(father), index_), 0) + 1
                
        return code
    code = []
    special_grammer = []
    asdl_node_to_code(asdl_ast_node, grammar, None, o_code, [0])
    #print(special_grammer)

    not_none_code = []
    for c in code:
        if c:
            not_none_code.append(c)
    return  not_none_code #' '.join(not_none_code)


def tree_sitter_ast_statistics(root):
    if len(root.children) == 0:
        return 0, 1, 1
    else:
        non_leaf_count_sum = 1 if len(root.children) > 0 else 0
        leaf_count_sum = 0
        max_height = 0
        for child_node in root.children:
            non_leaf_count, leaf_count, height = tree_sitter_ast_statistics(child_node)
            non_leaf_count_sum += non_leaf_count
            leaf_count_sum += leaf_count
            max_height = max(max_height, height+1)

        return  non_leaf_count_sum, leaf_count_sum, max_height

def plot_statistics(point_list, title, x, y):
    pass

def find_bpe_list(leaf, bpe_leaves, tokenizer, begin=0):
    end = begin+1
    find = 0
    while(end<=len(bpe_leaves)):
        if(leaf.strip() == tokenizer.decode(bpe_leaves[begin:end], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()):
            find = 1
            break
        else:
            end += 1
    if find == 0:
        return -1
    return end

def ast_to_code(cursor, list_no_win=None):
    def ast_node_to_code(ast_node):
        if not ast_node:
            # print('asdl_node is None')
            return code

        # argument_list -> None(( ( identifier identifier , , assignment_expression assignment_expression ) ))
        father_str = str(ast_node.type) + ' -> None(' + ' '.join([str(c.type) + ' ' + str(c.type) for c in ast_node.children]) + ')'
        print(father_str)
        for index_f, child in enumerate(ast_node.children):
            if child.child_count > 0:
                ast_node_to_code(child)
            else:
                if type(child.text) == list:
                    code.extend(child.text)
                else:
                    code.append(child.text)
                if list_no_win is not None:
                    if (child.type, father_str, index_f) not in list_no_win:
                        if code[-1] not in '{ } ( ) = . , >= [ ] != + / - -- == > < || ! ++ <= ? : && @ += ... * |= % /= & -= >>> >> << ~ | >>= &= -> ^ *= ^= >>>= :: ;'.split(' '):
                            code.append(' ')
        return code
    code = []
    ast_node_to_code(cursor, list_no_win)
    not_none_code = []
    for c in code:
        if c:
            not_none_code.append(c)
    return  not_none_code

def re_organize_code(lang, grammar, hyp, tokenizer):
    # JAVA_LANGUAGE = Language('build/my-languages.so', 'java' if lang == 'java' else 'c_sharp')
    # parser = Parser()
    # parser.set_language(JAVA_LANGUAGE)

    if os.path.exists('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+lang+'_list_no_win.bin'):
        # load list_no_win
        list_no_win = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+lang+'_list_no_win.bin', 'rb'))
    else:
        # load dict_blank from bin/dict_blank.bin
        dict_blank = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+lang+'_dict_blank.bin', 'rb'))
        # load dict_ from bin/dict.bin
        dict_ = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+lang+'_dict.bin', 'rb'))
        list_no_win = []
        for key in dict_:
            if key not in dict_blank:
                list_no_win.append(key)
            else:
                if dict_[key] > dict_blank[key]:
                    list_no_win.append(key)
        # save list_no_win to bin/list_no_win.bin
        pickle.dump(list_no_win, open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+lang+'_list_no_win.bin', 'wb'))

    # re_codes = []
    # for index, code in enumerate(codes):
        # tree = parser.parse(bytes(code, "utf8"))
    # cursor = tree.walk()
    # example = check(grammar, tree, None, code, 'test', -1, None, None, list_no_win)
    code_parts = asdl_to_code(hyp.tree, grammar, list_no_win, tokenizer, mode='tensor')
    code_parts_strip_blank = []
    for c in code_parts:
        # 如果是空格
        if c == ' ':
            code_parts_strip_blank.append(c)
        else:
            code_parts_strip_blank.append(c.strip())
    re_code = ''.join(code_parts_strip_blank)
    # re_code = ast_to_code(cursor, list_no_win)
    re_code = re_code[9:].strip()
    re_code = re_code[:-1] if re_code.count('{') != re_code.count('}') and re_code[-1]=='}' else re_code
    # re_codes.append(re_code)

    return re_code

def run(args):

    # grammar = ASDLGrammar.from_text(open('D:/project/codeGenerate/tranX-master/asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    # transition_system = LambdaCalculusTransitionSystem(grammar)

    # train_set = load_dataset(transition_system, 'D:/project/codeGenerate/tranX-master/data/atis/train.txt')


    
    PY_LANGUAGE = Language('/home/sly/CG/build/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    # code = "driver.find_element_by_xpath(\"//p[@id, 'one']/following-sibling::p\")"
    # tree = parser.parse(bytes(code, "utf8"))
    # cursor = tree.walk()
    # print(cursor.node.children[0].children[0].children[1].children)
    # exit()

    grammar_txt = 'py_asdl.txt'
    root_path = "/home/sly/CG/CodeT5/asdl/lang/java/"#full_grammar/"
    grammar_txt = root_path + grammar_txt
    print('grammar_txt: ', grammar_txt)
    asdl_text = open(grammar_txt).read()
    grammar = ASDLGrammar.from_text(asdl_text, root_type='module')

    file_list = [
        '/home/sly/CG/CodeT5/data/conala/conala-corpus/conala-train.json',
        '/home/sly/CG/CodeT5/data/conala/conala-corpus/conala-test.json',
    ]
    new_file_list = [
        '/home/sly/CG/CodeT5/data/conala/conala-corpus/new/conala-train.json',
        '/home/sly/CG/CodeT5/data/conala/conala-corpus/new/conala-test.json',
    ]
    # elif args.task == 'translate' and args.sub_task == 'java-cs':
    #     file_list = ['/home/sly/CG/CodeT5/data/translate/train.java-cs.txt.cs', '/home/sly/CG/CodeT5/data/translate/valid.java-cs.txt.cs', '/home/sly/CG/CodeT5/data/translate/test.java-cs.txt.cs']
    #     new_file_list = ['/home/sly/CG/CodeT5/data/translate/train.java-cs.txt.cs', '/home/sly/CG/CodeT5/data/translate/valid.java-cs.txt.cs', '/home/sly/CG/CodeT5/data/translate/test.java-cs.txt.cs']
    print(file_list)
    print(new_file_list)
    out = open('log_test.out', 'w+')
    examples_list = []
    file_index_list = []

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    # vocab = tokenizer.get_vocab()
    if args.dict:
        dict_blank, dict_ = dict(), dict()
    else:
        dict_blank, dict_ = None, None

    # list_no_win exist
    if args.use_dict:
        if os.path.exists('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_list_no_win.bin'):
            # load list_no_win
            list_no_win = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_list_no_win.bin', 'rb'))
        else:
            # load dict_blank from bin/dict_blank.bin
            dict_blank = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_dict_blank.bin', 'rb'))
            # load dict_ from bin/dict.bin
            dict_ = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_dict.bin', 'rb'))
            list_no_win = []
            for key in dict_:
                if key not in dict_blank:
                    list_no_win.append(key)
                else:
                    if dict_[key] > dict_blank[key]:
                        list_no_win.append(key)
            # save list_no_win to bin/lang_list_no_win.bin
            pickle.dump(list_no_win, open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_list_no_win.bin', 'wb'))
        # if os.path.exists('/home/sly/CG/CodeT5/asdl/lang/java/bin/list_blank_win.bin'):
        #     # load list_no_win
        #     list_no_win = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/list_blank_win.bin', 'rb'))
        # else:
        #     # load dict_blank from bin/dict_blank.bin
        #     dict_blank = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/dict_blank.bin', 'rb'))
        #     # load dict_ from bin/dict.bin
        #     dict_ = pickle.load(open('/home/sly/CG/CodeT5/asdl/lang/java/bin/dict.bin', 'rb'))
        #     list_no_win = []
        #     for key in dict_:
        #         if key not in dict_blank:
        #             list_no_win.append(key)
        #         else:
        #             if dict_[key] > dict_blank[key]:
        #                 list_no_win.append(key)
        #     # save list_no_win to bin/list_no_win.bin
        #     pickle.dump(list_no_win, open('/home/sly/CG/CodeT5/asdl/lang/java/bin/list_blank_win.bin', 'wb'))
    else:
        list_no_win = None
    
    for file_name, new_file_name in zip(file_list, new_file_list):
        # file = open(file_name)
        file = open(file_name, 'r')
        json_list = json.load(file)
        new_file = open(new_file_name, 'w')
        # data = [file.readlines()[0]]
        # re_organize_code('java', grammar, data[:10])
        # exit()
        # print('file_name: {}, line: {}'.format(file_name, len(data)))
        # continue
        file.close()
        examples = []
        index_list = []
        non_leaf_sum = []
        src_sum = []
        leaf_sum = []
        node_sum = []
        height_sum = []
        max_length = 0
        new_json_list = []
        # data = [data[index] for index in [17909, 20496, 21180, 22277, 24056]]
        # 1110, 2435, 3617, 4189, 6087, 9248, 9724, 9826, 10524, 10768, 14437, 16703
        for index, line in enumerate(tqdm(json_list)):
            # translate中是这样的
            # public ListSpeechSynthesisTasksResult listSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request) {request = beforeClientExecution(request);return executeListSpeechSynthesisTasks(request);}
            # public UpdateJourneyStateResult updateJourneyState(UpdateJourneyStateRequest request) {request = beforeClientExecution(request);return executeUpdateJourneyState(request);}
            code = line['snippet']
            # code = code.replace('\'', '"')
            # else:
                # code = json_line['code']
            tree = parser.parse(bytes(code, "utf8"))
            cursor = tree.walk()
            non_leaf_num, leaf_num, height = tree_sitter_ast_statistics(cursor.node)
            non_leaf_sum.append(non_leaf_num)
            leaf_sum.append(leaf_num)
            height_sum.append(height)
            if non_leaf_num+leaf_num > max_length:
                max_length = non_leaf_num+leaf_num
            example = check(grammar, cursor, out, code, file_name, index, dict_blank, dict_, list_no_win, tokenizer)
            if not example:
                print(index)
                continue
            # 删除example['tgt_canonical_code']开头的'class c { '和结尾的" }"
            code = example['tgt_canonical_code'].strip()
            tree_tgt_code = example['tree_tgt_code'].strip()
            write_line = tree_tgt_code
            new_json_line = line
            new_json_line['snippet'] = write_line
            new_json_list.append(new_json_line)

            if args.match_index:
                student_bpe = tokenizer.encode(code)
                teacher_bpe = tokenizer.encode('class c { ' + code + ' }')[1:-1]
                # find fitst match teacher index, and the index should add 1 cause there is always '</s>' following 'c'
                match_id = tokenizer.encode(' ' + code.split(' ')[1])[1:-1]
                match_index = []
                # get the index of match_id in teacher_bpe/student_bpe
                match_index.append(teacher_bpe.index(match_id[0]))
                match_index.append(student_bpe.index(match_id[0]))
                index_list.append(match_index)
            if not args.tree_label:
                continue
            begin = 0
            if example is not None:
                bpe_leaves = tokenizer.encode(example['tgt_canonical_code'])[1:-1]
                # src_query_tokens = json.loads(line)['nl']
                tgt_actions = example['tgt_actions']
                # 过滤GenToken[]
                tgt_actions = [action for action in tgt_actions if str(action) != 'GenToken[]']
                # tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions)  # 设置frontier的一些东西

                # example = Example(idx=index,
                #                   src_sent=src_query_tokens,
                #                   tgt_actions=tgt_action_infos,
                #                   tgt_code=code,
                #                   tgt_ast=None,  # asdl树
                #                   )

                app_rule_idx_row = []
                app_rule_mask_row = []
                token_row = []
                gen_token_mask_row = []
                for action_idx, action in enumerate(tgt_actions):
                    app_rule_idx = app_rule_mask = token_idx = gen_token_mask = 0
                    if isinstance(action, ApplyRuleAction):
                        app_rule_idx = grammar.prod2id[action.production]
                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        app_rule_mask = 1
                    elif isinstance(action, ReduceAction):
                        app_rule_idx = len(grammar)
                        app_rule_mask = 1
                    else:
                        token = str(action.token)
                        gen_token_mask = 1
                        if token == '</s>':
                            app_rule_idx_row.append(app_rule_idx)
                            app_rule_mask_row.append(app_rule_mask)
                            token_row.append(tokenizer.get_vocab()['</s>'])
                            gen_token_mask_row.append(gen_token_mask)
                            continue
                        end = find_bpe_list(token, bpe_leaves, tokenizer, begin)
                        if end == -1:
                            if str(action) == 'GenToken[]':
                                print('true')
                            else:
                                print('false')
                            print(code)
                            # print(tgt_actions)
                            # print(action)
                            print(token)
                            print(tokenizer.decode(bpe_leaves[:begin], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                            print(tokenizer.decode(bpe_leaves[begin:begin+2], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                            print(bpe_leaves, begin)
                            raise IndexError('超出索引')
                        token_bpe = bpe_leaves[begin:end]
                        begin = end
                        # token_idx = vocab[action.token]

                    if isinstance(action, GenTokenAction):
                        for token_idx in token_bpe:
                            app_rule_idx_row.append(app_rule_idx)
                            app_rule_mask_row.append(app_rule_mask)
                            token_row.append(token_idx)
                            gen_token_mask_row.append(gen_token_mask)
                        # app_rule_idx_row.append(len(grammar)) # reduce()
                        # app_rule_mask_row.append(1)
                        # token_row.append(0)
                        # gen_token_mask_row.append(0)
                    else:
                        app_rule_idx_row.append(app_rule_idx)
                        app_rule_mask_row.append(app_rule_mask)
                        token_row.append(token_idx)
                        gen_token_mask_row.append(gen_token_mask)


                examples.append((app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row))

                node_sum.append(len(token_row))
                src = line['rewritten_intent'].strip() if line['rewritten_intent'] else line['intent'].strip()
                src_tokens = tokenizer.encode(src, add_special_tokens=False)
                src_sum.append(len(src_tokens))

        # print(dict_blank, dict_)
        # save dict
        # TypeError: keys must be str, int, float, bool or None, not tuple

        # write new_json_list to new_file
        json.dump(new_json_list, new_file)
        new_file.close()
        examples_list.append(examples)
        file_index_list.append(index_list)
        non_leaf_average = np.mean(non_leaf_sum)
        non_leaf_var = np.var(non_leaf_sum)
        leaf_average = np.mean(leaf_sum)
        leaf_var = np.var(leaf_sum)
        height_average = np.mean(height_sum)
        height_var = np.var(height_sum)
        # 每段代码leaf数量和non_leaf数量之和大于256的比例
        cut_number = ((np.array(non_leaf_sum) + np.array(leaf_sum)) > 256).sum()/len(non_leaf_sum)
        print(file_name, \
              'cut_number: ', cut_number)
        print(file_name, \
              'non_leaf_average:', non_leaf_average, \
              'non_leaf_var', non_leaf_var, \
              'leaf_average: ', leaf_average, \
              'leaf_var: ', leaf_var, \
              'height_average: ', height_average, \
              'height_var: ', height_var, \
              'max_length: ', max_length)
        print(file_name, \
              'non_leaf_coef:', math.sqrt(non_leaf_var)/non_leaf_average, \
              'leaf_coef: ', math.sqrt(leaf_var)/leaf_average, \
              'height_coef: ', math.sqrt(height_var)/height_average)
        # node_sum = [i+j for i, j in zip(leaf_sum, non_leaf_sum)]
        # 打印分布的bins
        node_sum_counter = Counter(src_sum)
        # sort
        node_sum_counter = sorted(node_sum_counter.items(), key=lambda x: x[0])
        # 打印累积分布
        node_sum_counter = np.array(node_sum_counter)
        node_sum_counter[:, 1] = np.cumsum(node_sum_counter[:, 1])
        node_sum_counter = node_sum_counter.tolist()
        # 转化为百分比
        node_sum_counter = [[x[0], x[1]/len(node_sum)] for x in node_sum_counter]
        print('node_sum_counter: ', node_sum_counter)
        

    exit()
    out.close()
    if args.dict:
        pickle.dump(dict_blank, open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_dict_blank.bin', 'wb'))
        pickle.dump(dict_, open('/home/sly/CG/CodeT5/asdl/lang/java/bin/'+args.lang+'_dict.bin', 'wb'))
    # 对训练集生成primitive_vocab
    # primitive_tokens = [map(lambda a: a.action.token,
    #                         filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
    #                     for e in examples_list[0]]

    # primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=10)

    task_suffix = ''
    if args.task == 'translate' and args.sub_task == 'cs-java':
        if args.lang == 'java':
            task_suffix = '_translate_cs-java_java'
        elif args.lang == 'cs':
            task_suffix = '_translate_cs-java_cs'
    elif args.task == 'translate' and args.sub_task == 'java-cs':
        task_suffix = '_translate_cs'

    if args.match_index:
        pickle.dump(file_index_list[0], open('/home/sly/CG/CodeT5/asdl/lang/java/bin/train_conala_match_index.bin', 'wb'))
        pickle.dump(file_index_list[1], open('/home/sly/CG/CodeT5/asdl/lang/java/bin/test_conala_match_index.bin', 'wb'))

    if args.tree_label:
        print("tree_label length: {}" .format(len(examples_list[0])))
        print("tree_label length: {}" .format(len(examples_list[1])))
        print('/home/sly/CG/CodeT5/asdl/lang/java/bin/train_conala.bin')
        pickle.dump(examples_list[0], open('/home/sly/CG/CodeT5/asdl/lang/java/bin/train_conala.bin', 'wb'))
        pickle.dump(examples_list[1], open('/home/sly/CG/CodeT5/asdl/lang/java/bin/test_conala.bin', 'wb'))


def check_grammar(args):
    JAVA_LANGUAGE = Language('build/my-languages.so', 'java' if args.lang == 'java' else 'c_sharp')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    grammar_txt = 'java_asdl.txt'
    if args.task == 'translate' and args.sub_task == 'cs-java':
        if args.lang == 'java':
            grammar_txt = 'java_asdl_translate.txt'
        elif args.lang == 'cs':
            grammar_txt = 'cs_asdl_translate.txt'
    asdl_text = open(grammar_txt).readlines()
    ids_list = []
    
    print(len(asdl_text))
    for id, line in enumerate(asdl_text[4:]):
        if 'ERROR' in line:
            ids_list.append(id)
    print(ids_list)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="")
    parser.add_argument('--match_index', action='store_true')
    parser.add_argument('--tree_label', action='store_true')
    parser.add_argument('--task', type=str, default='concode')
    parser.add_argument('--sub_task', type=str, default=None)
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--dict', action='store_true')
    parser.add_argument('--use_dict', action='store_true')

    args = parser.parse_args()
    run(args)
    # check_grammar(args)