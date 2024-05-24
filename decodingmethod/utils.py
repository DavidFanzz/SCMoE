import torch
from torch.nn import functional as F
import numpy as np
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessorList,
)


def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


@torch.no_grad()
def dola(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=512,
    repetition_penalty=1.2,
    mature_layer=None,
    base_layer=None,
    candidate_premature_layers=None,
    relative_top=0.1,
    eos_token_id=None,
    stopping_criteria=None,
    early_stop=False,
):
    """
    - k: top-k candidate words are selected, default 3
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    early_exit_layers = candidate_premature_layers + [mature_layer]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # print("model inputs:",model_inputs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        if early_exit_layers is not None:
            dict_outputs = {}
            # loss_dict = {}
            for i, early_exit_layer in enumerate(early_exit_layers):
                # print(outputs.hidden_states.shape)
                # print(early_exit_layer)
                logits = model.lm_head(outputs.hidden_states[early_exit_layer])
                dict_outputs[early_exit_layer] = logits

        if base_layer is not None:
            base_logits = dict_outputs[base_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]
            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3

            logits = final_logits - base_logits
            next_token_logits = logits
        else:
            # 1. Stacking all premature_layers into a new dimension
            stacked_premature_layers = torch.stack(
                [dict_outputs[i][:, -1, :] for i in candidate_premature_layers], dim=0
            )

            # 2. Calculate the softmax values for mature_layer and all premature_layers
            softmax_mature_layer = F.softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1
            )  # shape: (batch_size, num_features)
            softmax_premature_layers = F.softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 3. Calculate M, the average distribution
            M = 0.5 * (
                softmax_mature_layer[None, :, :] + softmax_premature_layers
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 4. Calculate log-softmax for the KL divergence
            log_softmax_mature_layer = F.log_softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1
            )  # shape: (batch_size, num_features)
            log_softmax_premature_layers = F.log_softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 5. Calculate the KL divergences and then the JS divergences
            kl1 = F.kl_div(
                log_softmax_mature_layer[None, :, :], M, reduction="none"
            ).mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

            # 6. Reduce the batchmean
            js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
            premature_layer = candidate_premature_layers[
                int(js_divs.argmax().cpu().item())
            ]
            premature_layer_dist[premature_layer] += 1

            base_logits = dict_outputs[premature_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]

            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3
            logits = final_logits - base_logits
            next_token_logits = logits
            # pre-process distribution
        import copy

        new_next_token_logits = copy.deepcopy(next_token_logits)
        new_next_token_logits = new_next_token_logits.to(input_ids.device)
        next_tokens_scores = processors(input_ids, new_next_token_logits)

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_tokens_scores[:, eos_token_id] = -float("inf")

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # fsd-vec
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )

        if unfinished_sequences.max() == 0 or step == max_new_tokens:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids

@torch.no_grad()
def contrastive_decoding(
    teacher_model,
    student_model,
    teacher_t,
    student_t,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    alpha=0.1,
    beta=0.5,
    stopping_criteria=None,
):
    # formulation of "CONTRASTIVE DECODING IMPROVES REASONING IN LARGE LANGUAGE MODELS"
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs_student = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs_student["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(teacher_model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = teacher_model.prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )
        outputs = teacher_model(
            **model_inputs, return_dict=True, output_hidden_states=True
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = student_model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = student_model(
            **model_inputs_student,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")

        # next_tokens_scores = next_token_scores - alpha * next_token_logits_student
        next_tokens = torch.argmax(cdlogits, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = teacher_model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=teacher_model.config.is_encoder_decoder,
        )
        model_kwargs_student = student_model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=student_model.config.is_encoder_decoder,
        )
    return input_ids