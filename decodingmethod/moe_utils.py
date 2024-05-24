import torch
from torch.nn import functional as F
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

@torch.no_grad()
def scmoe(
    model,
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
    teacher_routed_tok=[0, 1],
    teacher_num_experts_per_tok=2,
    student_routed_tok=[0],
    student_num_experts_per_tok=1,
):

    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs_student = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs_student["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=teacher_routed_tok,
            num_experts_per_tok=teacher_num_experts_per_tok,
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = model(
            **model_inputs_student,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=student_routed_tok,
            num_experts_per_tok=student_num_experts_per_tok,
        )
        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")

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

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        model_kwargs_student = model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
    return input_ids

@torch.no_grad()
def scmoe_with_sampling(
    model,
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
    teacher_routed_tok=[0, 1],
    teacher_num_experts_per_tok=2,
    student_routed_tok=[0],
    student_num_experts_per_tok=1,
):

    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs_student = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs_student["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=teacher_routed_tok,
            num_experts_per_tok=teacher_num_experts_per_tok,
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = model(
            **model_inputs_student,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=student_routed_tok,
            num_experts_per_tok=student_num_experts_per_tok,
        )
        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")
        
        cdscores = F.softmax(cdlogits, dim=-1)
        next_tokens = torch.multinomial(cdscores, num_samples=1).squeeze(-1)
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

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        model_kwargs_student = model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
    return input_ids