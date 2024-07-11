"""
copy from https://github.com/huggingface/transformers/blob/cee768d97e42c6fcf744ba4d2a4dc8a8e78da4c1/src/transformers/models/llama/modeling_llama.py#L1221C1-L1293C28
"""
import torch


def prepare_inputs_for_generation(
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=True,
    **kwargs,
):
    past_length = 0
    if past_key_values is not None:
        # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
        past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length(
        )
        max_cache_length = (
            torch.tensor(past_key_values.get_max_length(),
                         device=input_ids.device)
            if past_key_values.get_max_length() is not None
            else None
        )
        cache_length = past_length if max_cache_length is None else torch.min(
            max_cache_length, past_length)

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_length == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs = {"input_ids": input_ids.contiguous()}

    input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
    if cache_position is None:
        cache_position = torch.arange(
            past_length, past_length + input_length, device=input_ids.device)
    elif use_cache:
        cache_position = cache_position[-input_length:]

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
