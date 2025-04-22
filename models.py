import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          GenerationMixin)
from transformers.pytorch_utils import torch_int_div
import logging
from asdl.transition_system import ApplyRuleAction, GenTokenAction
from asdl.lang.java.java_transition_system import asdl_to_code
from asdl.asdl import ASDLType
from components.action_info import ActionInfo
from components.decode_hypothesis import DecodeHypothesis
from collections import OrderedDict
import javalang
import time
from asdl.lang.java.java_transition_system import re_organize_code

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    # if args.load_model_path is not None:
    #     logger.info("Reload model from {}".format(args.load_model_path))
    #     model.load_state_dict(torch.load(args.load_model_path))

    return config, model, tokenizer


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CloneModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CloneModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.LongTensor(1).fill_(0).to(args.device)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

# class TreeCodeT5(GenerationMixin):
class TreeCodeT5(nn.Module):
    def __init__(self, args, vocab, transition_system, src_transition_system=None):
        # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        super(TreeCodeT5, self).__init__()

        self.args = args
        self.vocab = vocab
        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        config, model, tokenizer = build_or_load_gen_model(args)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.src_embed = model.shared
        self.model_dim = model.model_dim
        self.gen_from_vocab_head = model.lm_head # config.d_model->config.vocab_size
        self.config = config
        self.tokenizer = tokenizer

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last two entry are the embeddings for Reduce action and decoder_start_action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 2, self.model_dim)
        nn.init.xavier_normal_(self.production_embed.weight.data)
        self.apply_rule_head = nn.Linear(config.d_model, len(transition_system.grammar) + 2, bias=False) # grammar + reduceAction + pad_rule

        if src_transition_system is not None and args.source_ast:
            self.src_production_embed = nn.Embedding(len(src_transition_system.grammar) + 2, self.model_dim)

        if vocab is not None:
            self.primitive_embed = nn.Embedding(len(vocab), self.model_dim)
            nn.init.xavier_normal_(self.primitive_embed.weight.data)
            # self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab)).zero_())
        else:
            self.primitive_embed = model.shared
            # self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(model.shared.num_embeddings).zero_())

        # bias for predicting ApplyConstructor and GenToken actions
        # self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        # F.linear(input, self.primitive_embed.weight, self.tgt_token_readout_b)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None, # 应该是action id，如果不换decoder的embed的话应该直接传进来action embed不能是id
        decoder_attention_mask = None, # 应该是action mask
        labels = None, # 应该是action id, 也有token id，要做好区分
        decoder_inputs_embeds=None
    ):
        app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = labels
        # print('source_ast: ', self.args.source_ast)
        if self.args.source_ast:
            src_rule, src_rule_mask, src_token, src_token_mask = input_ids
            zero_action_mask = torch.zeros_like(src_rule).masked_fill(~(src_rule_mask + src_token_mask).bool(), self.src_production_embed.num_embeddings - 1)
            # print('device of src_rule', src_rule.device)
            # print('device of zero_action_mask', zero_action_mask.device)
            input_embed = self.src_production_embed(src_rule + zero_action_mask).masked_fill(~(src_rule_mask+zero_action_mask).bool().unsqueeze(-1), 0) + \
                          self.primitive_embed(src_token).masked_fill(~src_token_mask.bool().unsqueeze(-1), 0)
            encoder_outputs = self.encode(input_embeds=input_embed, attention_mask=attention_mask)
        else:
            encoder_outputs = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        action_mask = ((app_rule_mask_row + gen_token_mask_row).bool())
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_inputs_embeds = self._shift_right(app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
        decoder_outputs = self.decode(decoder_attention_mask = action_mask, decoder_inputs_embeds = decoder_inputs_embeds,
                                      hidden_states = encoder_outputs[0], attention_mask = attention_mask)

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # 计算primitive vocab上的分布概率
        gen_from_vocab_logits = self.gen_from_vocab_head(sequence_output)
        gen_from_vocab_prob = F.softmax(gen_from_vocab_logits, dim = -1)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=token_row.unsqueeze(2)).squeeze(2)

        # 计算production vocab上的分布概率
        apply_rule_logits = self.apply_rule_head(sequence_output)
        apply_rule_prob = F.softmax(apply_rule_logits, dim = -1)
        app_rule_idx_row = app_rule_idx_row.masked_fill(~app_rule_idx_row.bool(), self.production_embed.num_embeddings - 1)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2, # 不止应该比其他子词概率大，还得比production大
                                           index=app_rule_idx_row.unsqueeze(2)).squeeze(2)

        action_prob = tgt_apply_rule_prob.log() * ~gen_token_mask_row.bool() + tgt_primitive_gen_from_vocab_prob.log() * ~app_rule_mask_row.bool()
        loss = -action_prob.mean() # -torch.mean(torch.sum(action_prob, dim = -1))#

        output = (gen_from_vocab_logits, apply_rule_logits, ) + decoder_outputs[1:] + (encoder_outputs, )
        return ((loss,) + output) if loss is not None else output


    def encode(self, input_ids=None, attention_mask=None, input_embeds=None):
        if input_embeds is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )

        return encoder_outputs


    def decode(self, decoder_input_ids=None, decoder_attention_mask=None, decoder_inputs_embeds=None, hidden_states=None, attention_mask=None):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        return decoder_outputs


    def generate(self, source_ids, attention_mask, beam_size = None, max_length = None, frontier_field2mask_dict = None, under_record = False, tokenizer=None, return_tok_k=False, args=None, tree_labels=None):
        # source_ids,
        # attention_mask = source_mask,
        # use_cache = True,
        # num_beams = args.beam_size, 10
        # early_stopping = args.task == 'summarize', False
        # max_length = args.max_target_length) 150
        args = self.args
        primitive_vocab = self.vocab
        device = self.args.device
        _done = torch.tensor([False for _ in range(args.eval_batch_size)], dtype=torch.bool, device=device)
        # app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = tree_labels
        # print(app_rule_idx_row)
        # print(app_rule_mask_row)
        # print(token_row)
        # print(gen_token_mask_row)

        with torch.no_grad():
            if self.args.source_ast:
                src_rule, src_rule_mask, src_token, src_token_mask = source_ids
                zero_action_mask = torch.zeros_like(src_rule).masked_fill(~(src_rule_mask + src_token_mask).bool(), self.src_production_embed.num_embeddings - 1)
                # print('device of src_rule', src_rule.device)
                # print('device of zero_action_mask', zero_action_mask.device)
                input_embed = self.src_production_embed(src_rule + zero_action_mask).masked_fill(~(src_rule_mask+zero_action_mask).bool().unsqueeze(-1), 0) + \
                            self.primitive_embed(src_token).masked_fill(~src_token_mask.bool().unsqueeze(-1), 0)
                encoder_outputs = self.encode(input_embeds=input_embed, attention_mask=attention_mask)
            else:
                encoder_outputs = self.encode(input_ids=source_ids, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs[0].repeat_interleave(beam_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)

            # batch_size * beam_size
            hyp_scores = torch.FloatTensor(args.eval_batch_size, beam_size).fill_(0).to(args.device)
            hyp_scores[:, 1:] = -1e9
            hyp_scores = hyp_scores.view((args.eval_batch_size * beam_size,))

            # For computing copy probabilities, we marginalize over tokens with the same surface form
            # `aggregated_primitive_tokens` stores the position of occurrence of each source token
            # aggregated_primitive_tokens = OrderedDict()
            # for token_pos, token in enumerate(source_ids): # 记录每个token出现的位置，copy的时候会用到
            #     aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

            t = 0
            # batch_size个DecodeHypothesis
            hypotheses = [DecodeHypothesis()] * args.eval_batch_size * beam_size
            completed_hypotheses = [BeamHypotheses(num_beams=beam_size) for _ in range(args.eval_batch_size)]

            # token_row = [] # 生成的token
            # rule_row = [] # 生成的rule
            beam_indice = torch.arange(args.eval_batch_size * beam_size, dtype=torch.long, device=device)
            beam_rule_id = torch.LongTensor(args.eval_batch_size * beam_size).fill_(self.production_embed.num_embeddings - 1).to(args.device)
            beam_token_id = torch.LongTensor(args.eval_batch_size * beam_size).fill_(0).to(args.device)
            decoder_inputs_embeds = None

            # 记录解码过程中是否有beam连续重复生成K次
            # repeat_index = torch.cuda.LongTensor(args.eval_batch_size, beam_size).fill_(0)
            # repeat_count = torch.cuda.LongTensor(args.eval_batch_size, beam_size).fill_(0)
            # 如果连续重复生成K次，就打印这个beam
            while t < args.max_target_length:
                inputs_t = self.production_embed(beam_rule_id).masked_fill((~(beam_token_id==0)).unsqueeze(-1), 0) + \
                    self.primitive_embed(beam_token_id).masked_fill((beam_token_id==0).unsqueeze(-1), 0)
                inputs_t = inputs_t.unsqueeze(1)
                decoder_inputs_embeds = inputs_t if decoder_inputs_embeds is None else torch.cat((decoder_inputs_embeds[beam_indice], inputs_t), 1)
                # print(decoder_inputs_embeds.shape, encoder_hidden_states.shape, attention_mask.shape)
                # repeat_index = repeat_index.view(-1)[beam_indice].reshape(-1, beam_size)
                # repeat_count = repeat_count.view(-1)[beam_indice].reshape(-1, beam_size)
                decoder_outputs = self.decoder(
                    input_ids=None,
                    attention_mask=None,
                    inputs_embeds=decoder_inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                )
                # if(t>149):
                #     print(hypotheses[0].actions)

                decoder_output = decoder_outputs[0][:, -1, :]
                if self.config.tie_word_embeddings:
                    # Rescale output before projecting on vocab
                    decoder_output = decoder_output * (self.model_dim ** -0.5)

                gen_from_vocab_logits = self.gen_from_vocab_head(decoder_output)
                gen_from_vocab_prob = F.softmax(gen_from_vocab_logits, dim=-1)
                # primitive_log_prob = torch.log(gen_from_vocab_prob)
                apply_rule_logits = self.apply_rule_head(decoder_output)
                apply_rule_prob = F.softmax(apply_rule_logits, dim=-1)
                # apply_rule_log_prob = torch.log(apply_rule_prob)
                # if args.no_copy:
                #     primitive_prob = gen_from_vocab_prob
                # else:
                #     # Variable(batch_size, src_sent_len)
                #     primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)
                #
                #     # Variable(batch_size, 2)
                #     primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)
                #
                #     # Variable(batch_size, primitive_vocab_size)
                #     primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob
                #
                #     # if src_unk_pos_list:
                #     #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10
                rule_token_prob = torch.cat([apply_rule_prob, gen_from_vocab_prob], dim=1)
                total_mask = torch.empty((beam_size*args.eval_batch_size, self.production_embed.num_embeddings+self.primitive_embed.num_embeddings), dtype=torch.bool).to(device)
                for hyp_id, hyp in enumerate(hypotheses): # field为None的时候
                    key = str(hyp.frontier_field.type) + (str(hyp._value_buffer) if hyp.frontier_field.target_value is not None else 'None') if hyp.tree is not None else 'None'
                    if len(hyp._value_buffer) > 5:
                        rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                        token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                        token = self.tokenizer.get_vocab()['</s>']
                        token_mask[token] = 1
                        mask_value = torch.cat((rule_mask, token_mask), dim=0).bool()
                        # total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                        total_mask[hyp_id] = mask_value
                        continue
                    if key in frontier_field2mask_dict and not under_record:
                        mask_value = frontier_field2mask_dict[key]
                        # total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                        total_mask[hyp_id] = mask_value
                        continue

                    # generate new continuations
                    if not hyp.tree:
                        productions = self.grammar._production_ids[ASDLType(self.grammar.root_type)]
                        # productions = [productions[-1]]
                        if args.task == 'concode':
                            productions = [productions[-1]]
                        elif args.task == 'translate' and args.sub_task == 'cs-java':
                            # error_prods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1636, 2276, 2278, 2712]
                            productions = [i for i in productions if i not in self.grammar.error_ids]
                        elif args.task == 'translate' and args.sub_task == 'java-cs':
                            pass
                        rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                        rule_mask[productions] = 1
                        token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                    elif self.grammar.is_composite_type(hyp.frontier_field.type):
                        productions = self.grammar._production_ids[hyp.frontier_field.type]
                        if args.task == 'translate' and args.sub_task == 'cs-java':
                            # error_prods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1636, 2276, 2278, 2712]
                            productions = [i for i in productions if i not in self.grammar.error_ids]
                        rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                        rule_mask[productions] = 1
                        token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                    else:
                        # GenToken action
                        rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                        if hyp.frontier_field.target_value is not None: # 如果是none就只能靠模型自己去生成'</s>'
                            token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                            if len(hyp.frontier_field.target_value) == 1:
                                # 直接gentoken(hyp.frontier_field.target_value[0])
                                token = hyp.frontier_field.target_value[0]
                                token_mask[token] = 1
                            else:
                                if len(hyp._value_buffer) < len(hyp.frontier_field.target_value):
                                    token = hyp.frontier_field.target_value[len(hyp._value_buffer)]
                                    token_mask[token] = 1
                                else:
                                    # gentoken('</s>'), whose id = 2
                                    token = self.tokenizer.get_vocab()['</s>']
                                    token_mask[token] = 1
                        else:
                            token_mask = torch.ones(self.primitive_embed.num_embeddings).to(device)
                            token_mask[289] = 0 # 289是}的id
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                    rule_mask[self.grammar.error_ids] = 0
                    mask_i = torch.cat((rule_mask, token_mask), dim=0).bool()
                    # total_mask = mask_i.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_i.unsqueeze(0)), 0)
                    total_mask[hyp_id] = mask_i
                    # key not in frontier_field2mask_dict or under_record
                    frontier_field2mask_dict[key] = mask_i
                        
                        # if args.no_copy is False:
                        #     for token, token_pos_list in aggregated_primitive_tokens.items():
                        #         sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0,
                        #                                      Variable(T.LongTensor(token_pos_list))).sum()
                        #         gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob
                        #
                        #         if token in primitive_vocab:
                        #             token_id = primitive_vocab[token]
                        #             primitive_prob[hyp_id, token_id] = primitive_prob[
                        #                                                    hyp_id, token_id] + gated_copy_prob
                        #
                        #             hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                        #         else:
                        #             hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                        #                                       'copy_prob': gated_copy_prob.data.item()})

                        # if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                        #     unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                        #     token = hyp_unk_copy_info[unk_i]['token']
                        #     primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                        #     gentoken_new_hyp_unks.append(token)
                        #
                        #     hyp_copy_info[token] = (
                        #     hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

                new_hyp_scores = hyp_scores.unsqueeze(1) + torch.log(rule_token_prob * total_mask)
                new_hyp_scores = new_hyp_scores.view(args.eval_batch_size, -1)

                top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=2*beam_size, dim=1, largest=True, sorted=True) # 从所有的新hyp中选出topK个概率最高的
                next_indices = torch_int_div(top_new_hyp_pos, self.production_embed.num_embeddings + self.config.vocab_size)
                next_tokens = top_new_hyp_pos % (self.production_embed.num_embeddings + self.config.vocab_size)
                top_rule_mask = next_tokens <= (self.production_embed.num_embeddings-1)
                top_token_mask = next_tokens > (self.production_embed.num_embeddings-1)
                rule_id = next_tokens.masked_fill(top_rule_mask == False, self.production_embed.num_embeddings-1)
                token_id = (next_tokens - self.production_embed.num_embeddings).masked_fill(top_token_mask == False, 0)

                new_hypotheses = []
                for batch_index in range(args.eval_batch_size):
                    if _done[batch_index]:  # 在这里对每个batch进行关注，如果已经生成好了，就不用再生成了
                        if beam_size < len(completed_hypotheses[batch_index]):
                            raise ValueError(
                                f"Batch can only be done if at least {self.num_beams} beams have been generated")
                        # pad the batch
                        new_hypotheses.extend(hypotheses[batch_index * beam_size:(batch_index+1) * beam_size])
                        hyp_scores[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                        beam_indice[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                        beam_rule_id[batch_index * beam_size:(batch_index+1) * beam_size] = self.production_embed.num_embeddings - 1
                        beam_token_id[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                        continue

                    beam_num = 0
                    for beam_index in range(2*beam_size):
                        # if app_rule_mask_row[batch_index, t] == 0 and gen_token_mask_row[batch_index, t] == 0:
                        #     _done[batch_index] = True
                        #     break
                        action_info = ActionInfo()
                        # it's an ApplyRule or Reduce action
                        prev_hyp = hypotheses[batch_index * beam_size + next_indices[batch_index, beam_index].item()]
                        action = GenTokenAction(token_id[batch_index, beam_index]) if top_token_mask[batch_index, beam_index] != 0 \
                                    else ApplyRuleAction(self.grammar.id2prod[rule_id[batch_index, beam_index].item()])
                        # action = GenTokenAction(token_row[batch_index, t]) if gen_token_mask_row[batch_index, t] == 1 \
                        #             else ApplyRuleAction(self.grammar.id2prod[app_rule_idx_row[batch_index, t].item()])
                        # print('t: {}, batch: {}, beam: {}, action: {}'.format(t, batch_index, beam_index, action))
                        # copy_info = gentoken_copy_infos[k]

                        # if token_id == self.tokenizer.get_vocab()['<unk>']:
                        #     if gentoken_new_hyp_unks:
                        #         token = gentoken_new_hyp_unks[k]
                        #     else:
                        #         token = '<unk>'
                        # else:
                            # token = self.tokenizer.decoder[token_id.item()]

                        # if token in aggregated_primitive_tokens:
                        #     action_info.copy_from_src = True
                        #     action_info.src_token_position = aggregated_primitive_tokens[token]

                        action_info.action = action
                        action_info.t = t
                        if t > 0:
                            action_info.parent_t = prev_hyp.frontier_node.created_time
                            action_info.frontier_prod = prev_hyp.frontier_node.production
                            action_info.frontier_field = prev_hyp.frontier_field.field

                        new_hyp = prev_hyp.clone_and_apply_action_info(action_info) # 确认需不需要clone, 需要，原本的hyp不能直接改，后面的beam可能还要用到这个
                        new_hyp.score = top_new_hyp_scores[batch_index, beam_index]
                        # logger.info(new_hyp.frontier_field)
                        # logger.info(new_hyp.frontier_field_queue)

                        if new_hyp.completed:  # 什么时候会设置成完成：self.tree and self.frontier_field is None
                            # add length normalization
                            # new_hyp.score /= (t + 1)
                            is_beam_token_worse_than_top_num_beams = beam_index >= beam_size
                            if is_beam_token_worse_than_top_num_beams:
                                continue
                            completed_hypotheses[batch_index].add(new_hyp, new_hyp.score)
                        else:
                            new_hypotheses.append(new_hyp)
                            beam_indice[batch_index * beam_size + beam_num] = batch_index * beam_size + next_indices[batch_index, beam_index]
                            beam_rule_id[batch_index * beam_size + beam_num] = rule_id[batch_index, beam_index] if token_id[batch_index, beam_index] == 0 else self.production_embed.num_embeddings-1
                            beam_token_id[batch_index * beam_size + beam_num] = token_id[batch_index, beam_index] if token_id[batch_index, beam_index] != 0 else 0
                            hyp_scores[batch_index * beam_size + beam_num] = new_hyp.score
                            index = token_id[batch_index, beam_index].item() if token_id[batch_index, beam_index] != 0 \
                                        else rule_id[batch_index, beam_index].item()
                            # if index == repeat_index[batch_index, beam_num].item():
                            #     repeat_count[batch_index, beam_num] += 1
                            # else:
                            #     repeat_index[batch_index, beam_num] = index
                            #     repeat_count[batch_index, beam_num] = 0
                            beam_num += 1

                        if beam_num == beam_size:
                            break

                    _done[batch_index] = _done[batch_index] or completed_hypotheses[batch_index].is_done(new_hyp_scores[batch_index].max().item(), t+1)

                hypotheses = new_hypotheses
                # 如果repeat_count中的最大值大于5
                # if repeat_count.max() > 4:
                #     print('attention!')
                #     # 获得最大值的坐标
                #     index = repeat_count.view(-1).argmax()
                #     print(index)
                #     print(hypotheses[index].actions)
                t += 1
                if _done.all():
                    break

        # finalize
        final_beam_scores = hyp_scores
        for batch_idx, beam_hyp in enumerate(completed_hypotheses):
            if _done[batch_idx]:
                continue

            for beam_id in range(beam_size):  # 只有整个batch完成了才会加到self._beam_hyps中？这里是强行将未满足完成条件的hyp加入, 其实应该在这里把树生成完善之后再加入（但是其实难以完善除非只差最后几个
                batch_beam_idx = batch_idx * beam_size + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                beam_hyp.add(new_hypotheses[batch_beam_idx], final_score)

        # completed_hypotheses.sort(key=lambda hyp: -hyp.score)# 默认升序，从小到大。概率越高，score越大，-score越小，排序越前;
        completed_trees = []
        for batch_index, beam_hyp in enumerate(completed_hypotheses):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: -x[0]) # x[0]存的是score
            # completed_trees.append([hyp[1] for hyp in sorted_hyps[:-1]])
            completed_trees.append([hyp[1] for hyp in (sorted_hyps[:1] if not return_tok_k else sorted_hyps)])
            # 如果sorted_hyps的个数不等于beam_size
            if len(sorted_hyps) != beam_size:
                # 抛出错误
                raise ValueError('hyp number not equal to beam_size')
        batch_completed_code = []
        for index in range(len(completed_trees)): # 正式版只翻译top1就行
            completed_code = []
            code = re_organize_code('java' if args.sub_task == 'cs-java' else 'cs', self.grammar, completed_trees[index][0], tokenizer)
            completed_code.append(code)
            # for tree in completed_trees[index]:
            #     one_topk = asdl_to_code(tree.tree, self.grammar)
            #     completed_code.append(one_topk)
            # if tokenizer is not None:
            #     top1_sentence = tokenizer.decode(completed_code[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #     try:
            #         tree = javalang.parse.parse(top1_sentence)
            #     except Exception as e:
            #         for tree, tokens in zip(completed_trees[index], completed_code):
            #             print('*****************************************************************************************')
            #             print(tree.actions)
            #             print(tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            batch_completed_code.append(completed_code if return_tok_k else completed_code[0])

        # print(rule_row, token_row)
        return batch_completed_code

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def _shift_right(self, app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        zero_action_mask = torch.zeros_like(app_rule_idx_row).masked_fill(~(app_rule_mask_row + gen_token_mask_row).bool(), self.production_embed.num_embeddings - 1)
        input_embed = self.production_embed(app_rule_idx_row + zero_action_mask).masked_fill(~(app_rule_mask_row+zero_action_mask).bool().unsqueeze(-1), 0) + \
                          self.primitive_embed(token_row).masked_fill(~gen_token_mask_row.bool().unsqueeze(-1), 0)
        shifted_input_embed = input_embed.new_zeros(input_embed.shape)
        shifted_input_embed[:, 1:, :] = input_embed[:, :-1, :].clone()
        shifted_input_embed[:, 0, :] = self.primitive_embed(torch.LongTensor(1).fill_(decoder_start_token_id).to(app_rule_idx_row.device))
        if self.args.rule_begin:
            shifted_input_embed[:, 0, :] = self.production_embed(torch.LongTensor(1).fill_(self.production_embed.num_embeddings - 1).to(app_rule_idx_row.device))

        # replace possible -100 values in labels by `pad_token_id`
        # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_embed

    def get_decoder_attention_mask(self, t, max_decoder_length):
        mask = torch.ones(max_decoder_length, max_decoder_length)
        mask = torch.tril(mask).bool()  # 下三角
        return mask[t]


class Classifier(nn.Module):
    def __init__(self, args, model_dim):
        super(Classifier, self).__init__()
        self.args = args
        self.model_dim = model_dim
        if self.args.complex_classifier:
            self.linear1 = nn.Linear(args.classifier_input_argument * model_dim, 512)
            self.linear2 = nn.Linear(512, 2)
        else:
            self.linear = nn.Linear(model_dim, 2)

    def forward(self, x):
        if self.args.complex_classifier:
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
        else:
            x = self.linear(x)

        return x
    

class TreeSeqCodeT5(TreeCodeT5):
    def __init__(self, args, vocab, transition_system, src_transition_system=None):
        super().__init__(args, vocab, transition_system)
        # super(TreeCodeT5, self).__init__()

        config, model, tokenizer = build_or_load_gen_model(args)
        # add special tokens <seq>/<tree>
        tokenizer.add_special_tokens({'additional_special_tokens': ['<seq>', '<tree>']})
        self.tree_token_id = tokenizer.convert_tokens_to_ids('<tree>')
        self.seq_token_id = tokenizer.convert_tokens_to_ids('<seq>')
        model.resize_token_embeddings(len(tokenizer))
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.src_embed = model.shared
        self.model_dim = model.model_dim
        self.gen_from_vocab_head = model.lm_head # config.d_model->config.vocab_size
        config.vocab_size = len(tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        if args.tune_on_label:
            self.decode_classifier = Classifier(args, self.model_dim) # 根据encoder最后一层hidden state的average来判断是seq还是tree

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last two entry are the embeddings for Reduce action and decoder_start_action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 2, self.model_dim)
        nn.init.xavier_normal_(self.production_embed.weight.data)
        self.apply_rule_head = nn.Linear(config.d_model, len(transition_system.grammar) + 2, bias=False) # grammar + reduceAction + pad_rule

        self.primitive_embed = model.shared


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        seq_labels = None,
        tree_labels=None,
        seq_decoder_attention_mask=None,
        tree_decoder_attention_mask=None,
        decode_type_label=None,
        first_token_loss_mask=None,
        class_ratio=1.0,
        seq_code_ids=None,
        tree_code_ids=None,
        example_weight=None
    ):
        app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row = tree_labels
        encoder_outputs = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        if self.args.tune_on_label and first_token_loss_mask is not None:
            # encoder最后一层hidden state的average pooling
            classifier_input_src = encoder_outputs[0].mean(dim=1)
            # classifier_input_src = self.primitive_embed(input_ids).mean(dim=1)
            classifier_input_seq_tgt = self.primitive_embed(seq_code_ids).mean(dim=1)
            classifier_input_tree_tgt = self.primitive_embed(tree_code_ids).mean(dim=1)
            if self.args.classifier_input_argument == 1:
                classifier_input = classifier_input_src
            elif self.args.classifier_input_argument == 3:
            # 前三者的拼接
                classifier_input = torch.cat([classifier_input_src, classifier_input_seq_tgt, classifier_input_tree_tgt], dim=-1)
            # 输入到分类器，得到输出后计算loss
            classifier_output = self.decode_classifier(classifier_input)
            if first_token_loss_mask.sum() > 0:
                class_weights = torch.tensor([1.0, 1.0]).to(classifier_input.device)
                if self.args.reweight:
                    class_weights = torch.tensor([1.0, class_ratio]).to(classifier_input.device)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                loss = loss_fn(classifier_output[first_token_loss_mask], decode_type_label[first_token_loss_mask])
                # example_weight * loss
                loss = (example_weight * loss).mean()
                if self.args.margin_loss:
                    loss_fn = nn.TripletMarginLoss(margin=self.args.margin, p=2)
                    # 根据batch中数量少的一类的数量，确定triplet的数量
                    triplet_num = min((decode_type_label == 0).sum(), (decode_type_label == 1).sum())
                    # 在batch范围内随机生成triplet_num个随机数
                    triplet_indices = torch.randperm(first_token_loss_mask.shape[0])[:triplet_num]
                    # 取出indices对应的样本作为锚点样本
                    anchors = classifier_input[triplet_indices]
                    # 获取锚点样本的label，确定对每个样本来说，正样本的label是什么
                    anchor_labels = decode_type_label[triplet_indices]
                    # loss = 0
                    for i in range(triplet_num):
                        anchor_label = anchor_labels[i]
                        triplet_indice = triplet_indices[i]
                        # 为每个锚点样本采样一个正样本(避免采样到锚点样本本身)和负样本
                        positive_mask = decode_type_label == anchor_label
                        positive_mask[triplet_indice] = False
                        if positive_mask.sum() == 0:
                            continue
                        positive = classifier_input[positive_mask][torch.randperm(positive_mask.sum())[0]]
                        negative_mask = decode_type_label != anchor_label
                        negative = classifier_input[negative_mask][torch.randperm(negative_mask.sum())[0]]
                        # 计算triplet loss
                        anchor = anchors[i]
                        loss += loss_fn(anchor, positive, negative) * class_weights[anchor_label]
            else:
                loss = torch.tensor(0., requires_grad=True).to(classifier_input.device)
            return (loss, )

        # get tree decoder inputs from shifting lm labels to the right
        tree_decoder_inputs_embeds = self._shift_right(app_rule_idx_row, app_rule_mask_row, token_row, gen_token_mask_row)
        # get seq deocoder inputs
        # temp = torch.zeros_like(seq_labels)
        # temp[..., :-1] = seq_labels[..., 1:]
        # seq_labels = temp
        # temp_mask = torch.zeros_like(seq_decoder_attention_mask)
        # temp_mask[..., :-1] = seq_decoder_attention_mask[..., 1:]
        # seq_decoder_attention_mask = temp_mask
        seq_decoder_inputs_embeds = self._shift_right_seq(seq_labels)
        decoder_inputs_embeds = torch.where(decode_type_label.unsqueeze(1).unsqueeze(1), seq_decoder_inputs_embeds, tree_decoder_inputs_embeds)
        decoder_attention_mask = torch.where(decode_type_label.unsqueeze(1), seq_decoder_attention_mask, tree_decoder_attention_mask)
        decoder_outputs = self.decode(decoder_attention_mask = decoder_attention_mask, decoder_inputs_embeds = decoder_inputs_embeds,
                                      hidden_states = encoder_outputs[0], attention_mask = attention_mask)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)
        
        # 计算primitive vocab上的分布概率
        gen_from_vocab_logits = self.gen_from_vocab_head(sequence_output)
        gen_from_vocab_prob = F.softmax(gen_from_vocab_logits, dim = -1)
        tree_tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=token_row.unsqueeze(2)).squeeze(2)
        seq_tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=seq_labels.unsqueeze(2)).squeeze(2)
        if self.args.count_decode_label:
            pred_decode_label = gen_from_vocab_prob[:, 0].argmax(dim=-1)
            limit_pred_decode_label = gen_from_vocab_prob[:, 0, -2:].argmax(dim=-1) + 32100
            
        # 计算production vocab上的分布概率
        apply_rule_logits = self.apply_rule_head(sequence_output)
        apply_rule_prob = F.softmax(apply_rule_logits, dim = -1)
        app_rule_idx_row = app_rule_idx_row.masked_fill(~app_rule_idx_row.bool(), self.production_embed.num_embeddings - 1)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2, # 不止应该比其他子词概率大，还得比production大
                                           index=app_rule_idx_row.unsqueeze(2)).squeeze(2)

        # action_prob = tgt_apply_rule_prob.log() * ~gen_token_mask_row.bool() + tree_tgt_primitive_gen_from_vocab_prob.log() * gen_token_mask_row.bool()
        action_prob = tgt_apply_rule_prob.log() * app_rule_mask_row.bool() + tree_tgt_primitive_gen_from_vocab_prob.log() * gen_token_mask_row.bool()
        
        action_mask = ((app_rule_mask_row + gen_token_mask_row).bool())
        if first_token_loss_mask is None:
            # 统一不计算第一个位置的loss
            action_prob = action_prob[:, 1:]
            action_mask = action_mask[:, 1:]
            seq_token_prob = seq_tgt_primitive_gen_from_vocab_prob[decode_type_label].log()[:, 1:]
        else:
            action_prob[:, 0] = action_prob[:, 0] * first_token_loss_mask
            action_mask[:, 0] = first_token_loss_mask
            seq_token_prob = seq_tgt_primitive_gen_from_vocab_prob[decode_type_label].log()
            seq_token_prob[:, 0] = seq_token_prob[:, 0] * first_token_loss_mask[decode_type_label]

        loss_tree = torch.sum(action_prob[~decode_type_label])/torch.sum(action_mask[~decode_type_label]) if action_prob[~decode_type_label].sum() else torch.tensor(0).to(action_prob.device)

        loss_seq = seq_token_prob.mean() if seq_token_prob.sum() else torch.tensor(0).to(action_prob.device)
        loss = -(loss_tree+loss_seq).mean()
        # 如果loss等于nan
        if torch.isnan(loss):
            # 打印loss_tree、loss_seq
            logger.info('loss_tree: {}'.format(loss_tree))
            logger.info('loss_seq: {}'.format(loss_seq))
            # 打印tgt_apply_rule_prob.log() * app_rule_mask_row、tgt_primitive_gen_from_vocab_prob.log() * gen_token_mask_row
            logger.info('encoder_outputs[0]: {}'.format(encoder_outputs[0]))
            logger.info('decoder_input_embeds: {}'.format(decoder_inputs_embeds))
            logger.info('sequence_output: {}'.format(sequence_output))
            logger.info('gen_from_vocab_logits: {}'.format(gen_from_vocab_logits))
            logger.info('apply_rule_logits: {}'.format(apply_rule_logits))
            logger.info('gen_from_vocab_prob: {}'.format(gen_from_vocab_prob))
            logger.info('seq_tgt_primitive_gen_from_vocab_prob: {}'.format(seq_tgt_primitive_gen_from_vocab_prob))
            logger.info('******************************************************')

        output = (gen_from_vocab_logits, apply_rule_logits, ) + decoder_outputs[1:] + (encoder_outputs, )
        return ((tuple([loss, -loss_seq, -loss_tree, pred_decode_label, limit_pred_decode_label])) + output) if self.args.count_decode_label \
            else ((tuple([loss, -loss_seq, -loss_tree])) + output)
    

    def _shift_right_seq(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        shifted_input_embeds = self.primitive_embed(shifted_input_ids)
        if self.args.rule_begin:
            shifted_input_embeds[:, 0] = self.production_embed(torch.LongTensor(1).fill_(self.production_embed.num_embeddings - 1).to(input_ids.device))

        return shifted_input_embeds
    

    def generate(self, source_ids, attention_mask, beam_size = None, max_length = None, frontier_field2mask_dict = None,
                 under_record = False, tokenizer=None, return_tok_k=False, decode_type=None, seq_code_ids=None, tree_code_ids=None):
        args = self.args
        device = self.args.device
        _done = torch.tensor([False for _ in range(args.eval_batch_size)], dtype=torch.bool, device=device)

        if args.unlimit_decode:
            return self.unlimit_decode(source_ids, attention_mask, beam_size, max_length, frontier_field2mask_dict, under_record, tokenizer, return_tok_k, decode_type)
        
        with torch.no_grad():
            encoder_outputs = self.encode(input_ids=source_ids, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs[0].repeat_interleave(beam_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)

            if args.tune_on_label:
                classifier_input_src = encoder_outputs[0].mean(dim=1)
                # classifier_input_src = self.primitive_embed(source_ids).mean(dim=1)
                classifier_input_seq_tgt = self.primitive_embed(seq_code_ids).mean(dim=1)
                classifier_input_tree_tgt = self.primitive_embed(tree_code_ids).mean(dim=1)
                if self.args.classifier_input_argument == 1:
                    classifier_input = classifier_input_src
                elif self.args.classifier_input_argument == 3:
                    classifier_input = torch.cat([classifier_input_src, classifier_input_seq_tgt, classifier_input_tree_tgt], dim=-1)
                classifier_output = self.decode_classifier(classifier_input)
                classifier_output_label = torch.argmax(F.softmax(classifier_output), dim=1)
                return classifier_output_label, F.softmax(classifier_output)

            # batch_size * beam_size
            hyp_scores = torch.FloatTensor(args.eval_batch_size, beam_size).fill_(0).to(args.device) # 记录每个beam的得分
            hyp_scores[:, 1:] = -1e9 # 除了第一个beam，其他beam的得分都是-inf
            hyp_scores = hyp_scores.view((args.eval_batch_size * beam_size,))
            decode_type_label = torch.ones(args.eval_batch_size, dtype=torch.bool, device=device) # 记录每个batch的解码类型

            t = 0
            # batch_size个DecodeHypothesis
            hypotheses = [[] for _ in range(args.eval_batch_size * beam_size)] # 记录每个解码步骤的结果，先按照树的方式初始化，如果是序列则会替换为list
            completed_hypotheses = [BeamHypotheses(num_beams=beam_size) for _ in range(args.eval_batch_size)] # 记录每个batch完成的解码结果

            beam_indice = torch.arange(args.eval_batch_size * beam_size, dtype=torch.long, device=device) # 记录当前topk的beam是上一步的哪些beam扩展得到
            beam_rule_id = torch.LongTensor(args.eval_batch_size * beam_size).fill_(self.production_embed.num_embeddings - 1).to(args.device) # 记录当前步是哪些rule(如果预测的是rule结点)
            beam_token_id = torch.LongTensor(args.eval_batch_size * beam_size).fill_(0).to(args.device) # 记录当前步是哪些token(如果预测的是token结点, 对序列解码来说所有预测都是token结点)
            decoder_inputs_embeds = None

            # with torch.profiler.profile(
            #     schedule=torch.profiler.schedule(
            #         wait=2,
            #         warmup=2,
            #         active=6,
            #         repeat=1),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler('tensorboard/trace_expand'),
            # ) as profiler:
            if args.speed_optimize:
                generate_time = 0
                process_time = 0
                start_time = time.time()
                current_time = 0
            if True:
                while t < args.max_target_length:
                    if not args.rule_begin:
                        inputs_t = self.production_embed(beam_rule_id).masked_fill((~(beam_token_id==0)).unsqueeze(-1), 0) + \
                            self.primitive_embed(beam_token_id).masked_fill((beam_token_id==0).unsqueeze(-1), 0) # token_begin
                    else:
                        inputs_t = self.production_embed(beam_rule_id).masked_fill(((beam_rule_id==self.production_embed.num_embeddings - 1)).unsqueeze(-1), 0) + \
                        self.primitive_embed(beam_token_id).masked_fill(~(beam_rule_id==self.production_embed.num_embeddings - 1).unsqueeze(-1), 0) # rule_begin
                    inputs_t = inputs_t.unsqueeze(1)
                    decoder_inputs_embeds = inputs_t if decoder_inputs_embeds is None else torch.cat((decoder_inputs_embeds[beam_indice], inputs_t), 1)
                    decoder_outputs = self.decoder(
                        input_ids=None,
                        attention_mask=None,
                        inputs_embeds=decoder_inputs_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=attention_mask,
                    )

                    decoder_output = decoder_outputs[0][:, -1, :]
                    if self.config.tie_word_embeddings:
                        # Rescale output before projecting on vocab
                        decoder_output = decoder_output * (self.model_dim ** -0.5)

                    gen_from_vocab_logits = self.gen_from_vocab_head(decoder_output)
                    gen_from_vocab_prob = F.softmax(gen_from_vocab_logits, dim=-1)
                    # primitive_log_prob = torch.log(gen_from_vocab_prob)
                    apply_rule_logits = self.apply_rule_head(decoder_output)
                    apply_rule_prob = F.softmax(apply_rule_logits, dim=-1)
                    rule_token_prob = torch.cat([apply_rule_prob, gen_from_vocab_prob], dim=1)
                    total_mask = None # change to while, jump seq query
                    if t==0:
                        key = 'tree_seq_mask'
                        if key in frontier_field2mask_dict:
                            mask_value = frontier_field2mask_dict[key]
                            total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                        else:
                            # 除了<seq><tree>对应位置，其他位置都置零
                            rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                            token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                            token_mask[self.seq_token_id] = 1
                            token_mask[self.tree_token_id] = 1
                            mask_value = torch.cat((rule_mask, token_mask), dim=0).bool()
                            total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                            # 把相关mask矩阵存入字典，'tree_seq_mask': mask_value
                            frontier_field2mask_dict[key] = mask_value
                        total_mask = mask_value.unsqueeze(0).expand(args.eval_batch_size * beam_size, -1)
                    else:
                    #     mask_value = frontier_field2mask_dict['seq_mask']
                    #     total_mask = mask_value.unsqueeze(0).expand(args.eval_batch_size * beam_size, -1)
                        hyp_id = 0
                        while hyp_id<len(hypotheses):
                            if decode_type_label[hyp_id//beam_size]==True: # 序列解码
                                key = 'seq_mask'
                                if key in frontier_field2mask_dict:
                                    mask_value = frontier_field2mask_dict[key]
                                else:
                                    # 使用只保留token概率的mask
                                    rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                                    token_mask = torch.ones(self.primitive_embed.num_embeddings).to(device)
                                    token_mask[-2:] = 0
                                    mask_value = torch.cat((rule_mask, token_mask), dim=0).bool()
                                    # 把相关mask矩阵存入字典，'seq_mask': mask_value
                                    if key not in frontier_field2mask_dict:
                                        frontier_field2mask_dict[key] = mask_value
                                total_mask = mask_value.unsqueeze(0).expand(beam_size, -1) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0).expand(beam_size, -1)), 0)
                                hyp_id += beam_size
                                continue
                            
                            hyp = hypotheses[hyp_id]
                            key = str(hyp.frontier_field.type) + (str(hyp._value_buffer) if hyp.frontier_field.target_value is not None else 'None') if hyp.tree is not None else 'None'
                            if len(hyp._value_buffer) > 5:
                                rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                                token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                                token = self.tokenizer.get_vocab()['</s>']
                                token_mask[token] = 1
                                mask_value = torch.cat((rule_mask, token_mask), dim=0).bool()
                                total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                                hyp_id += 1
                                continue
                            if key in frontier_field2mask_dict and not under_record:
                                mask_value = frontier_field2mask_dict[key]
                                # 在最后两维补False
                                if mask_value.shape[0] == self.production_embed.num_embeddings + self.primitive_embed.num_embeddings - 2:
                                    mask_value = torch.cat((mask_value, torch.zeros(2).bool().to(device)), dim=0)
                                    frontier_field2mask_dict[key] = mask_value
                                total_mask = mask_value.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_value.unsqueeze(0)), 0)
                                hyp_id += 1
                                continue

                            # generate new continuations
                            if not hyp.tree:
                                productions = self.grammar._production_ids[ASDLType(self.grammar.root_type)]
                                productions = [productions[-1]]
                                rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                                for prod_id in productions:
                                    rule_mask[prod_id] = 1
                                token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                            elif self.grammar.is_composite_type(hyp.frontier_field.type):
                                productions = self.grammar._production_ids[hyp.frontier_field.type]
                                rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                                for prod_id in productions:
                                    rule_mask[prod_id] = 1
                                token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                            else:
                                # GenToken action
                                rule_mask = torch.zeros(self.production_embed.num_embeddings).to(device)
                                if hyp.frontier_field.target_value is not None: # 如果是none就只能靠模型自己去生成'</s>'
                                    token_mask = torch.zeros(self.primitive_embed.num_embeddings).to(device)
                                    if len(hyp.frontier_field.target_value) == 1:
                                        # 直接gentoken(hyp.frontier_field.target_value[0])
                                        token = hyp.frontier_field.target_value[0]
                                        token_mask[token] = 1
                                    else:
                                        if len(hyp._value_buffer) < len(hyp.frontier_field.target_value):
                                            token = hyp.frontier_field.target_value[len(hyp._value_buffer)]
                                            token_mask[token] = 1
                                        else:
                                            # gentoken('</s>'), whose id = 2
                                            token = self.tokenizer.get_vocab()['</s>']
                                            token_mask[token] = 1
                                else:
                                    token_mask = torch.ones(self.primitive_embed.num_embeddings).to(device)
                                    token_mask[289] = 0 # 289是}的id
                                hyp_copy_info = dict()  # of (token_pos, copy_prob)
                                hyp_unk_copy_info = []

                            mask_i = torch.cat((rule_mask, token_mask), dim=0).bool()
                            total_mask = mask_i.unsqueeze(0) if total_mask==None else torch.cat((total_mask, mask_i.unsqueeze(0)), 0)
                            # key not in frontier_field2mask_dict or under_record
                            frontier_field2mask_dict[key] = mask_i
                            hyp_id += 1

                    new_hyp_scores = hyp_scores.unsqueeze(1) + torch.log(rule_token_prob * total_mask)
                    new_hyp_scores = new_hyp_scores.view(args.eval_batch_size, -1)

                    top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=2*beam_size, dim=1, largest=True, sorted=True) # 从所有的新hyp中选出topK个概率最高的
                    next_indices = torch_int_div(top_new_hyp_pos, self.production_embed.num_embeddings + self.config.vocab_size)
                    next_tokens = top_new_hyp_pos % (self.production_embed.num_embeddings + self.config.vocab_size)
                    top_rule_mask = next_tokens <= (self.production_embed.num_embeddings-1)
                    top_token_mask = next_tokens > (self.production_embed.num_embeddings-1)
                    rule_id = next_tokens.masked_fill(top_rule_mask == False, self.production_embed.num_embeddings-1)
                    token_id = (next_tokens - self.production_embed.num_embeddings).masked_fill(top_token_mask == False, 0)
                    if args.speed_optimize:
                        current_time = time.time()
                        generate_time += current_time - start_time
                        start_time = current_time

                    new_hypotheses = []
                    for batch_index in range(args.eval_batch_size):
                        if _done[batch_index]:  # 在这里对每个batch进行关注，如果已经生成好了，就不用再生成了
                            if beam_size < len(completed_hypotheses[batch_index]):
                                raise ValueError(
                                    f"Batch can only be done if at least {self.num_beams} beams have been generated")
                            # pad the batch
                            new_hypotheses.extend(hypotheses[batch_index * beam_size:(batch_index+1) * beam_size])
                            hyp_scores[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                            beam_indice[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                            beam_rule_id[batch_index * beam_size:(batch_index+1) * beam_size] = self.production_embed.num_embeddings - 1
                            beam_token_id[batch_index * beam_size:(batch_index+1) * beam_size] = 0
                            continue

                        if t==0:
                            if self.args.tune_on_label:
                                first_token = self.seq_token_id if classifier_output_label[batch_index].item() else self.tree_token_id
                            elif decode_type is None:
                                first_token = token_id[batch_index, 0].item()
                            elif decode_type == 'tree':
                                first_token = self.tree_token_id
                            else:
                                first_token = self.seq_token_id
                            # 如果t==0，那么不需要找beam_size个分支，全部用第一个分支
                            for beam_num in range(beam_size):
                                if first_token==self.seq_token_id:
                                    new_hypotheses.append([])
                                else:
                                    new_hypotheses.append(DecodeHypothesis())
                            beam_indice[batch_index * beam_size:(batch_index+1) * beam_size] = batch_index * beam_size + next_indices[batch_index, 0]
                            beam_rule_id[batch_index * beam_size:(batch_index+1) * beam_size] = self.production_embed.num_embeddings-1
                            # beam_token_id[batch_index * beam_size:(batch_index+1) * beam_size] = token_id[batch_index, 0]
                            beam_token_id[batch_index * beam_size:(batch_index+1) * beam_size] = first_token
                            # 更新decode_type_labels
                            decode_type_label[batch_index] = True if first_token==self.seq_token_id else False
                            continue
                        beam_num = 0
                        for beam_index in range(2*beam_size):
                            prev_hyp = hypotheses[batch_index * beam_size + next_indices[batch_index, beam_index]]
                            # 如果是tree就是一棵tree，否则是一个list
                            if decode_type_label[batch_index]: # 序列解码
                                new_hyp = prev_hyp[:]
                                new_hyp.append(token_id[batch_index, beam_index])
                                batch_beam_idx = batch_index * beam_size + beam_num
                                if (self.config.eos_token_id is not None) and (token_id[batch_index, beam_index].item() == self.config.eos_token_id):
                                    # if beam_token does not belong to top num_beams tokens, it should not be added
                                    is_beam_token_worse_than_top_num_beams = beam_index >= beam_size
                                    if is_beam_token_worse_than_top_num_beams:
                                        continue
                                    
                                    completed_hypotheses[batch_index].add(
                                        new_hyp, # 没存input_ids, tree是直接存在树里的
                                        top_new_hyp_scores[batch_index, beam_index],
                                    )
                                else:
                                    # add next predicted token since it is not eos_token
                                    # next_beam_scores[batch_idx, beam_idx] = next_score
                                    # next_beam_tokens[batch_idx, beam_idx] = next_token
                                    # next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                                    new_hypotheses.append(new_hyp)
                                    beam_indice[batch_beam_idx] = batch_index * beam_size + next_indices[batch_index, beam_index]
                                    beam_rule_id[batch_beam_idx] = self.production_embed.num_embeddings-1
                                    beam_token_id[batch_beam_idx] = token_id[batch_index, beam_index]
                                    hyp_scores[batch_beam_idx] = top_new_hyp_scores[batch_index, beam_index]
                                    beam_num += 1

                            else: # 树解码
                                action_info = ActionInfo()
                                # it's an ApplyRule or GenToken action
                                action = GenTokenAction(token_id[batch_index, beam_index]) if top_token_mask[batch_index, beam_index] \
                                            else ApplyRuleAction(self.grammar.id2prod[rule_id[batch_index, beam_index].item()])

                                action_info.action = action
                                action_info.t = t
                                if t > 1:
                                    action_info.parent_t = prev_hyp.frontier_node.created_time
                                    action_info.frontier_prod = prev_hyp.frontier_node.production
                                    action_info.frontier_field = prev_hyp.frontier_field.field

                                new_hyp = prev_hyp.clone_and_apply_action_info(action_info) # 确认需不需要clone, 需要，原本的hyp不能直接改，后面的beam可能还要用到这个
                                new_hyp.score = top_new_hyp_scores[batch_index, beam_index]

                                if new_hyp.completed:  # 什么时候会设置成完成：self.tree and self.frontier_field is None
                                    is_beam_token_worse_than_top_num_beams = beam_index >= beam_size
                                    if is_beam_token_worse_than_top_num_beams:
                                        continue
                                    completed_hypotheses[batch_index].add(new_hyp, new_hyp.score)
                                else:
                                    new_hypotheses.append(new_hyp)
                                    beam_indice[batch_index * beam_size + beam_num] = batch_index * beam_size + next_indices[batch_index, beam_index]
                                    beam_rule_id[batch_index * beam_size + beam_num] = rule_id[batch_index, beam_index] if token_id[batch_index, beam_index] == 0 else self.production_embed.num_embeddings-1
                                    beam_token_id[batch_index * beam_size + beam_num] = token_id[batch_index, beam_index] if token_id[batch_index, beam_index] != 0 else 0
                                    hyp_scores[batch_index * beam_size + beam_num] = new_hyp.score
                                    beam_num += 1

                            if beam_num == beam_size:
                                break

                        _done[batch_index] = _done[batch_index] or completed_hypotheses[batch_index].is_done(new_hyp_scores[batch_index].max().item(), t+1)
                        if args.speed_optimize:
                            current_time = time.time()
                            process_time += current_time - start_time
                            start_time = current_time

                    hypotheses = new_hypotheses
                    t += 1
                    if _done.all():
                        break
                    # profiler.step()

        # finalize
        final_beam_scores = hyp_scores
        for batch_idx, beam_hyp in enumerate(completed_hypotheses):
            if _done[batch_idx]:
                continue

            for beam_id in range(beam_size):  # 只有整个batch完成了才会加到self._beam_hyps中？这里是强行将未满足完成条件的hyp加入, 其实应该在这里把树生成完善之后再加入（但是其实难以完善除非只差最后几个
                batch_beam_idx = batch_idx * beam_size + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                beam_hyp.add(new_hypotheses[batch_beam_idx], final_score)

        # completed_hypotheses.sort(key=lambda hyp: -hyp.score)# 默认升序，从小到大。概率越高，score越大，-score越小，排序越前;
        completed_trees = []
        for batch_index, beam_hyp in enumerate(completed_hypotheses):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: -x[0]) # x[0]存的是score
            # completed_trees.append([hyp[1] for hyp in sorted_hyps[:-1]])
            completed_trees.append([hyp[1] for hyp in (sorted_hyps[:1] if not return_tok_k else sorted_hyps)])
            # 如果sorted_hyps的个数不等于beam_size
            # if len(sorted_hyps) != beam_size:
            #     # 抛出错误
            #     raise ValueError('hyp number not equal to beam_size')
        batch_completed_code = []
        batch_completed_tree = []
        for index in range(len(completed_trees)): # 正式版只翻译top1就行
            completed_code = []
            if not decode_type_label[index]: # tree
                for tree in completed_trees[index]:
                    one_topk = asdl_to_code(tree.tree, self.grammar)
                    completed_code.append(one_topk)
            else:
                completed_code = completed_trees[index]
            batch_completed_code.append(completed_code if return_tok_k else completed_code[0])
            batch_completed_tree.append(completed_trees[0])

        if args.speed_optimize:
            g_minute = int((generate_time % 3600) // 60)
            logger.info('generate time: {}m {}s'.format(g_minute, int(generate_time % 60)))
            p_minute = int((process_time % 3600) // 60)
            logger.info('process time: {}m {}s'.format(p_minute, int(process_time % 60)))
        return batch_completed_code, decode_type_label, t


    def generate_sample(self, source_ids, attention_mask, beam_size = None, max_length = None, frontier_field2mask_dict = None, 
                 under_record = False, tokenizer=None, return_tok_k=False, decode_type=None, seq_code_ids=None, tree_code_ids=None):
        pass


    def unlimit_decode(self, source_ids, attention_mask, beam_size = None, max_length = None, frontier_field2mask_dict = None, under_record = False, tokenizer=None, return_tok_k=False, decode_type=None):
        pass


class BeamHypotheses():
    def __init__(self, num_beams, length_penalty=1.0, early_stopping=False):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = -1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, beam_indices = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams: # 如果超过了beam size, 则删除最差的那个
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else: # 如果没有超过beam size, 加完更新最低分后就不管了
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty # 长度惩罚在这里
            ret = self.worst_score >= cur_score
            return ret
