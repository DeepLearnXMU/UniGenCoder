# coding=utf-8

from .asdl import *
from .asdl_ast import AbstractSyntaxTree
from .transition_system import *


class Hypothesis(object):
    def __init__(self):
        self.tree = None
        self.actions = []
        self.score = 0.
        self.frontier_node = None
        self.frontier_field = None
        self._value_buffer = []
        self.frontier_field_queue = []

        # record the current time step
        self.t = 0
        self.None_field = None

    def __len__(self):
        return len(self.actions)

    def apply_action(self, action):
        if self.tree is None:
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding'

            self.tree = AbstractSyntaxTree(action.production)
            # # 将tree中的fields反序加入frontier_field_queue
            # for field in self.tree.fields[::-1]:
            #     self.frontier_field_queue.append(field)
            self.update_frontier_info()
        elif self.frontier_node:
            if isinstance(self.frontier_field.type, ASDLCompositeType) and action: # or self.frontier_field.type.name == "ERROR"
                if isinstance(action, ApplyRuleAction):
                    field_value = AbstractSyntaxTree(action.production)
                    field_value.created_time = self.t
                    self.frontier_field.add_value(field_value)
                    # # 将frontier_node中的fields反序加入frontier_field_queue
                    # # print("value: ", self.frontier_field.value)
                    # value = self.frontier_field.value[::-1] if isinstance(self.frontier_field.value, list) else [self.frontier_field.value]
                    # for field in value:
                    #     self.frontier_field_queue.append(field)
                    self.update_frontier_info()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                        'applied on field with multiple ' \
                                                                                        'cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Invalid action [%s] on field [%s]' % (action, self.frontier_field))
            else:  # fill in a primitive field
                if isinstance(action, GenTokenAction):
                    # only field of type string requires termination signal </primitive> --> <\s>
                    end_primitive = False
                    # if self.frontier_field.type.name == 'string':
                    if self.frontier_field.target_value is None or len(self.frontier_field.target_value) > 1:
                        if action.is_stop_signal():
                            self.frontier_field.add_value(self._value_buffer)
                            self._value_buffer = []

                            end_primitive = True
                        else:
                            self._value_buffer.append(action.token)
                    else:
                        self.frontier_field.add_value(action.token)
                        end_primitive = True



                        # if str(self.frontier_field) == 'Field(== ==)':
                        #     print(action)
                        #     print(self.frontier_field)


                    if end_primitive and self.frontier_field.cardinality in ('single', 'optional'):
                        self.frontier_field.set_finish()
                        # # 将frontier_field_queue中满足条件的field弹出
                        # while len(self.frontier_field_queue) > 0:
                        #     # 是简单类型并且已完成
                        #     if isinstance(self.frontier_field_queue[-1], ASDLPrimitiveType) and self.frontier_field_queue[-1].finished:
                        #         self.frontier_field_queue.pop()
                        #     elif isinstance(self.frontier_field_queue[-1], ASDLCompositeType):
                        #         self.frontier_field_queue.pop()
                        #         break
                        #     else:
                        #         break
                        self.update_frontier_info()

                # elif isinstance(action, ReduceAction):
                #     assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                #                                                                         'applied on field with multiple ' \
                #                                                                         'cardinality'
                #     self.frontier_field.set_finish()
                #     self.update_frontier_info()
                # else:
                #     raise ValueError('Can only invoke GenToken or Reduce actions on primitive fields') # 存在当beam_size个hyp都只有一种有效action，且前beam_size个beam有completed，那么就会选中inf的action，这样的action必然是非法的，就会引起报错

        self.t += 1
        self.actions.append(action)

    def update_frontier_info(self):
        def _find_frontier_node_and_field(tree_node):
            if tree_node:
                for field in tree_node.fields:
                    # if it's an intermediate node, check its children
                    if isinstance(field.type, ASDLCompositeType) and field.value:
                        if field.cardinality in ('single', 'optional'): iter_values = [field.value]
                        else: iter_values = field.value

                        for child_node in iter_values:
                            result = _find_frontier_node_and_field(child_node)
                            if result: return result

                    # now all its possible children are checked
                    if not field.finished:
                        return tree_node, field

                return None
            else: return None

        frontier_info = _find_frontier_node_and_field(self.tree)
        if frontier_info:
            self.frontier_node, self.frontier_field = frontier_info
        else:
            self.frontier_node, self.frontier_field = None, None

    def clone_and_apply_action(self, action):
        new_hyp = self.copy()
        new_hyp.apply_action(action)

        return new_hyp

    def copy(self):
        new_hyp = Hypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t

        new_hyp.update_frontier_info()

        return new_hyp

    @property
    def completed(self):
        return self.tree and self.frontier_field is None
