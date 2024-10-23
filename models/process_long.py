import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens, structural_mask):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)#转为Tensor放入cuda
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 500:
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            decoder_attention_mask=attention_mask,
            structural_mask=torch.stack(structural_mask, dim=0),
            return_dict=True,
        )
        attention = output.encoder_attentions[-1]
        decoder_context = output.encoder_last_hidden_state
        context_outputs = output.last_hidden_state
    else:
        new_input_ids, new_attention_mask, num_seg, new_structure_mask = [], [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 500:
                new_input_ids.append(input_ids[i, :500])
                new_attention_mask.append(attention_mask[i, :500])
                num_seg.append(1)
                new_structure_mask.append(structural_mask[i])
            else:
                input_ids1 = torch.cat([input_ids[i, :500 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 500 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :500]
                attention_mask2 = attention_mask[i, (l_i - 500): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                new_structure_mask.extend([structural_mask[i], structural_mask[i]])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        structural_mask = torch.stack(new_structure_mask, dim=0)
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            decoder_attention_mask=attention_mask,
            structural_mask=structural_mask,
            return_dict=True,
        )
        attention = output.encoder_attentions[-1]
        decoder_context = output.encoder_last_hidden_state
        context_outputs = output.last_hidden_state
        i = 0
        new_decoder_context, new_context_outputs, new_attention = [], [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output_decoder_context = F.pad(decoder_context[i], (0, 0, 0, c - 500))
                output_context_outputs = F.pad(context_outputs[i], (0, 0, 0, c - 500))
                att = F.pad(attention[i], (0, c - 500, 0, c - 500))
                new_decoder_context.append(output_decoder_context)
                new_context_outputs.append(output_context_outputs)
                new_attention.append(att)
            elif n_s == 2:
                output_decoder_context_1 = decoder_context[i][:500 - len_end]
                output_context_outputs_1 = context_outputs[i][:500 - len_end]
                mask1 = attention_mask[i][:500 - len_end]
                att1 = attention[i][:, :500 - len_end, :500 - len_end]
                output_decoder_context_1 = F.pad(output_decoder_context_1, (0, 0, 0, c - 500 + len_end))
                output_context_outputs_1 = F.pad(output_context_outputs_1, (0, 0, 0, c - 500 + len_end))
                mask1 = F.pad(mask1, (0, c - 500 + len_end))
                att1 = F.pad(att1, (0, c - 500 + len_end, 0, c - 500 + len_end))

                output_decoder_context_2 = decoder_context[i + 1][len_start:]
                output_context_outputs_2 = context_outputs[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output_decoder_context_2 = F.pad(output_decoder_context_2, (0, 0, l_i - 500 + len_start, c - l_i))
                output_context_outputs_2 = F.pad(output_context_outputs_2, (0, 0, l_i - 500 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 500 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 500 + len_start, c - l_i, l_i - 500 + len_start, c - l_i])

                mask = mask1 + mask2 + 1e-10
                output_decoder_context = (output_decoder_context_1 + output_decoder_context_2) / mask.unsqueeze(-1)
                output_context_outputs = (output_context_outputs_1 + output_context_outputs_2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_attention.append(att)
                new_decoder_context.append(output_decoder_context)
                new_context_outputs.append(output_context_outputs)
            i += n_s
        decoder_context = torch.stack(new_decoder_context, dim=0)
        context_outputs = torch.stack(new_context_outputs, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return decoder_context, context_outputs, attention


def process_long_input_decode(model, input_ids, attention_mask, start_tokens, end_tokens, enc_attention_mask, enc_hidden):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)#转为Tensor放入cuda
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 500:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_attention_mask,
            output_attentions=True
        )
        sequence_output = output.last_hidden_state
    else:
        new_input_ids, new_enc_ids, new_attention_mask, num_seg, new_enc_mask = [], [], [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 500:
                new_input_ids.append(input_ids[i, :500])
                new_attention_mask.append(attention_mask[i, :500])
                new_enc_mask.append(enc_attention_mask[i])
                new_enc_ids.append(enc_hidden[i])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :500 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 500 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :500]
                attention_mask2 = attention_mask[i, (l_i - 500): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                new_enc_mask.extend([enc_attention_mask[i], enc_attention_mask[i]])
                new_enc_ids.extend([enc_hidden[i], enc_hidden[i]])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        enc_attention_mask = torch.stack(new_enc_mask, dim=0)
        enc_hidden = torch.stack(new_enc_ids, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_attention_mask,
            output_attentions=True
        )
        sequence_output = output.last_hidden_state
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 500))
                new_output.append(output)
            elif n_s == 2:
                output1 = sequence_output[i][:500 - len_end]
                mask1 = attention_mask[i][:500 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 500 + len_end))
                mask1 = F.pad(mask1, (0, c - 500 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 500 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 500 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                new_output.append(output)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)

    return sequence_output