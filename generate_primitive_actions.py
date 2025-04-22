from asdl.lang.java.java_transition_system import run, check_grammar
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="")
    parser.add_argument('--match_index', action='store_true')
    parser.add_argument('--tree_label', action='store_true')
    parser.add_argument('--task', type=str, default='translate')
    parser.add_argument('--sub_task', type=str, default='cs-java')
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--dict', action='store_true')
    parser.add_argument('--use_dict', action='store_true')

    args = parser.parse_args()
    run(args)
    # check_grammar(args)