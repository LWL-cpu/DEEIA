# DEEIA model
import torch
import torch.nn as nn
from models.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple
from opt_einsum import contract
from models.modeling_roberta_ import RobertaModel_, RobertaPreTrainedModel
from models.modeling_roberta_ import *
from models.process_long import process_long_input_decode, process_long_input
class DEEIA(RobertaPreTrainedModel):
    def __init__(self, config, decode_layer_start=17):
        super().__init__(config)
        self.config = config
        decode_layer_start = config.encoder_layers
        # self.model = BartModel(config, structural_mask=config.structural_type)
        self.roberta = RobertaModel_(config, decode_layer_start=decode_layer_start, structural_mask=config.structural_type)
        self.w_prompt_start = nn.Parameter(torch.rand(config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(config.hidden_size, ))
        self.decode_layer_start = decode_layer_start

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        self.contextual_merger = nn.Linear(2 * config.hidden_size, config.hidden_size)


    def reset(self):
        self.w_prompt_start = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.roberta._init_weights(self.contextual_merger.weight)

    def context_pooling(self, value_matrix, trigger_att, hidden_rep):  # Event_specific aggregation

        att = (value_matrix* trigger_att)
        att = att / (att.sum(1, keepdim=True) + 1e-5)  # 防止分母出现0
        rs = contract("ld,rl->rd", hidden_rep, att)

        return rs


    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        all_ids=None,
        all_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        arg_list=None,
        event_triggers=None,
        enc_attention_mask=None
    ):
        """
        Args:
            multi args post calculation
        """
        context_outputs_ = self.roberta(
            input_ids=all_ids,
            attention_mask=all_mask_ids,
            output_hidden_states=True,
            fully_encode=True,
            output_attentions=True,
            return_dict=True,
        structural_mask=torch.stack(enc_attention_mask, dim=0))  # DE module
        enc_outputs = context_outputs_.hidden_states
        decoder_context = enc_outputs[self.decode_layer_start]

        if self.config.context_representation == 'decoder':
            context_outputs = enc_outputs[-1]
        else:
            context_outputs = decoder_context

        encoder_attentions = context_outputs_.attentions[self.decode_layer_start].mean(1)
        encoder_attentions = encoder_attentions


        logit_lists = list()
        total_loss = 0.
        if len(event_triggers) == 0:
            print(len(event_triggers))
        for i, (context_output, encoder_attention, arg_joint_prompt, old_tok_to_new_tok_index, event_trigger) in \
            enumerate(zip(context_outputs, encoder_attentions, arg_joint_prompts, old_tok_to_new_tok_indexs, event_triggers)):
            
            batch_loss = list()
            cnt = 0
            list_output = list()
            for ii in range(len(event_trigger)):
                event_trigger_pos = event_trigger[ii]
                event_trigger_attention = torch.mean(encoder_attention[event_trigger_pos[0]:event_trigger_pos[1]], dim=0).unsqueeze(0)
            
                output = dict()
                for arg_role in arg_joint_prompt[ii].keys():
                    """
                    "arg_role": {"tok_s": , "tok_e": }
                    """
                    prompt_slots = arg_joint_prompt[ii][arg_role]

                    start_logits_list = list()
                    end_logits_list = list()
                    for (p_start,p_end, p_start_off, p_end_off) in zip(prompt_slots['tok_s'], prompt_slots['tok_e'], prompt_slots['tok_s_off'], prompt_slots['tok_e_off']):
                        prompt_query_sub = context_output[p_start_off:p_end_off]

                        prompt_query_sub_attention = encoder_attention[p_start_off:p_end_off]
                        prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                        prompt_query_sub_attention = torch.mean(prompt_query_sub_attention, dim=0).unsqueeze(0)
                        context_rs = self.context_pooling(prompt_query_sub_attention, event_trigger_attention, decoder_context[i])
                        prompt_query_sub = torch.tanh(self.contextual_merger(torch.cat((prompt_query_sub, context_rs), dim=-1)))

                        start_query = (prompt_query_sub*self.w_prompt_start).unsqueeze(-1) # [1, H, 1]
                        end_query = (prompt_query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)

                    output[arg_role] = [start_logits_list, end_logits_list]

                    if self.training:
                        # calculate loss
                        target = target_info[i][ii][arg_role] # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans)<len(predicted_spans):
                            # need to consider whether to make more
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0,0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len

                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss)/2)
                list_output.append(output)
            logit_lists.append(list_output)
            if self.training: # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt
            
        if self.training:
            return total_loss/len(context_outputs), logit_lists
        else:
            return [], logit_lists

