#!/usr/bin/env python
# coding: utf-8

# In[1]:
from dataclasses import dataclass
from typing import List
import numpy as np
import torchaudio
from vocos import Vocos
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
import time
import gc
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
from modelscope import snapshot_download
from transformers.models.llama import LlamaForCausalLM
import os
import resource

from einops import rearrange
from dvae import DVAE
from gpt import GPT_warpper
import torch
# 挑选最适合的设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型下载
model_dir = snapshot_download('mirror013/ChatTTS')

model_config = {
    'num_audio_tokens': 626,
    'num_text_tokens': 21178,
    'gpt_config': {
        'hidden_size': 768,
        'intermediate_size': 3072,
        'num_attention_heads': 12,
        'num_hidden_layers': 20,
        'use_cache': False,
        'max_position_embeddings': 4096,
        'spk_emb_dim': 192,
        'spk_KL': False,
        'num_audio_tokens': 626,
        'num_text_tokens': None,
        'num_vq': 4
    }
}

dvae_config = {
    'dim': 512,
    'decoder_config': {
        'idim': 512,
        'odim': 512,
        'n_layer': 12,
        'bn_dim': 128,
    },
    'vq_config': {
        'dim': 1024,
        'levels': [5, 5, 5, 5],
        'G': 2,
        'R': 2,
    }
}
SEED = 1397
# 加载模型
model = GPT_warpper(**model_config).to(device).eval()
model.load_state_dict(torch.load(os.path.join(
    model_dir, "asset/GPT.pt"), map_location='cpu'))

decoder = DVAE(**dvae_config).to(device).eval()
decoder.load_state_dict(torch.load(os.path.join(
    model_dir, "asset/DVAE.pt"), map_location='cpu'))

# 加载分词器
tokenizer = torch.load(os.path.join(
    model_dir, "asset/tokenizer.pt"), map_location='cpu')
tokenizer.padding_side = 'left'

# 设置参数
top_k = 50
top_p = 0.7
temperature = 1e-8
max_length = 2048
text = "接下来,杨叔，借我看一下现场地图。他肯定穿过了前面的那扇门，不可能在这么小的地方晃悠了两小时。"

print("all load done.")
def resource_usage_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        iteration_count = 0

        # 执行被修饰的函数
        result = func(*args, **kwargs)

        for output in result:
            iteration_count += 1
            yield output

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 获取最大内存使用量（以字节为单位）
        max_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Max Memory Usage: {max_memory_usage / 1024} MB")

        # 获取用户模式和系统模式下的CPU时间
        user_time, sys_time = resource.getrusage(resource.RUSAGE_SELF)[:2]
        print(f"User Mode CPU Time: {user_time} seconds")
        print(f"System Mode CPU Time: {sys_time} seconds")

        # 如果CUDA设备可用，打印CUDA内存使用情况
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            print(
                f"CUDA Memory Usage: {used_memory / 1024**2} MB used out of {total_memory / 1024**2} MB")

        # 打印函数调用时间和迭代次数
        print(f"Function {func.__name__} executed in {elapsed_time} seconds")
        if iteration_count > 0:
            print(
                f"Iterations per second: {iteration_count / elapsed_time} it/s")

    return wrapper


"""
copy from https://github.com/huggingface/transformers/blob/cee768d97e42c6fcf744ba4d2a4dc8a8e78da4c1/src/transformers/models/llama/modeling_llama.py#L1221C1-L1293C28
"""
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

# ## 准备input_ids等前置工作
@dataclass(repr=False, eq=False)
class AudioOutput():
    hiddens: torch.tensor
    ids: torch.tensor


# In[4]:
@resource_usage_decorator
@torch.no_grad()
def generate_with_sampling(prompt, max_new_tokens, **kwargs) -> List[AudioOutput]:

    if not isinstance(prompt, list):
        prompt = [prompt]

    # 添加必要的标签
    prompt = [f'[Stts][empty_spk]{i}[Ptts]' for i in prompt]

    text_token = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False, padding=True)

    # 分离开必要的信息
    input_ids = text_token['input_ids']
    attention_mask = text_token['attention_mask']
    token_type_ids = text_token['token_type_ids']

    # 初始化 logits processors
    logits_processor = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        RepetitionPenaltyLogitsProcessor(penalty=1.05),
        TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=3),
        TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=3),
    ])

    # cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

    past_key_values = DynamicCache()

    text_mask = torch.ones(
        input_ids.shape, dtype=bool, device=device)

    # 第一次使用需要初始化emb
    # 仅在第一次需要生效而已
    input_ids = input_ids[..., None].expand(-1, -1, model.num_vq)
    input_emb = model.get_emb(input_ids, text_mask,
                              attention_mask=attention_mask)
    start_idx = input_ids.shape[1]
    # 记录结束的token
    eos_token_id = model.emb_code[0].num_embeddings - 1

    # 用于存储注意力掩码
    attention_mask_cache = torch.ones(  # 创建一个全为 1 的张量
        (input_ids.shape[0],  # 表示批次大小(batch size)。
            input_ids.shape[1]+max_new_tokens,),  # 表示输入序列的长度加上最大新生成的 token 数，以便容纳输入序列和生成的token
        dtype=torch.bool,
        device=input_ids.device)
    attention_mask_cache[:, :attention_mask.shape[1]
                         ] = attention_mask  # 第一次应用需要将已有的进行标记
    cache_position = None
    for i in range(max_new_tokens):
        # 构建传入的生成参数
        model_input = prepare_inputs_for_generation(input_ids=input_ids,
                                                    # inputs_embeds=input_emb,
                                                    past_key_values=past_key_values,
                                                    attention_mask=attention_mask_cache[:,
                                                                                        :input_ids.shape[1]],
                                                    use_cache=True,
                                                    cache_position=cache_position,
                                                    )

        if i != 0:
            # 因为采用了自制的多头注意力机制
            # 需要将input_ids 转换成input_embeding方式
            # 并且要注意emb_code是将音频token转化成向量，所以之前的文本token是不适用的，一定要裁剪掉
            code_emb = [model.emb_code[i](
                model_input['input_ids'][:, :, i]) for i in range(model.num_vq)]
            input_emb = torch.stack(code_emb, 3).sum(3)

        model_input['input_ids'] = None
        model_input['inputs_embeds'] = input_emb
        cache_position = model_input['cache_position']

        # 前向传播
        outputs: BaseModelOutputWithPast = model.gpt.forward(**model_input,
                                                             return_dict=True,
                                                             output_attentions=True,
                                                             output_hidden_states=True)

        """一般普通的 LLM 主要处理单一模态的数据，如纯文本。它们通常只有一个输出头部（例如，一个线性层），用于将模型的隐藏状态映射到词汇表的 logits。
        在多模态生成任务中，模型需要处理不同类型的输入数据（如文本和音频），并生成相应的输出。因此，需要多个输出头部来分别处理不同模态的数据。例如，self.head_text 用于处理文本数据，self.head_code 用于处理音频数据。
        并且音频数据的维度对应着不同的码本,之后将这些logits沿着一个新的维度堆叠并求和，以整合不同量化层的信息。
        这里是从last_hidden_state也就是预测出来的隐藏状态通过num_vq个音频线性层转换成num_vq个token
        """
        logits = torch.stack([model.head_code[i](outputs.last_hidden_state) for i in range(
            model.num_vq)], 3)    # 形状为 (batch_size, audio_token, emb_output_dim, num_vq)
        # 提取每个序列的最后一个 token 的 logits
        logits = logits[:, -1].float()

        logits = rearrange(logits, "b c n -> (b n) c")
        # logits = logits.permute(0, 2, 1).view(-1, logits.size(1))

        # 预测不带上输入的文本避免污染音频生成
        logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
        # logits_token = input_ids[:, start_idx:].permute(0, 2, 1).view(-1, input_ids[:, start_idx:].size(1))

        # 这里是基于输入的文本进行的预测，而不包括后面生成的token
        logits = logits_processor(logits_token, logits)

        # 使用多项分布采样下一个 token
        scores = torch.nn.functional.softmax(logits, dim=-1)

        # 得出下一个预测的token (audio_token, batch_size)
        next_tokens = torch.multinomial(scores, num_samples=1)

        # idx_next 重新排列为形状 (audio_token, num_vq)
        idx_next = next_tokens.view(-1, model.num_vq)

        # 更新生成的序列
        input_ids = torch.cat([input_ids, idx_next.unsqueeze(1)], 1)

        # 更新cache_position位置
        cache_position = model.gpt._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_input)['cache_position']

        # 检查任意是否生成了结束标记
        # 注意！这里只是为了简便才这么做，需要根据不同的batch的情况来具体分析才对
        if (eos_token_id == next_tokens).any(0).item():
            break

        past_key_values = outputs.past_key_values
        # 形状为(b, num_vq)
        yield AudioOutput(ids=idx_next, hiddens=outputs.last_hidden_state[:, -1])    # 直接返回下一个token，可以用做dvae进行解码处理


torch.manual_seed(SEED)
import torch

def process_audio_tokens(prompt, max_length, batch_size=24):
    audio_mel_spec = []
    audio_token = []
    hiddens = []

    for iter, audio_output in enumerate(generate_with_sampling(prompt=prompt, max_new_tokens=max_length)):
        audio_token.append(audio_output.ids)
        hiddens.append(audio_output.hiddens)
        
        # 集齐 batch_size 个 audio token 后再一起解析
        if (iter + 1) % batch_size == 0:
            audio_mel_spec.append(
                decoder(torch.cat(audio_token, dim=0)[None].permute(0, 2, 1))
            )
            audio_token.clear()

    # 处理剩余的 audio token
    if audio_token:
        audio_mel_spec.append(
            decoder(torch.cat(audio_token, dim=0)[None].permute(0, 2, 1))
        )

    return audio_mel_spec,hiddens 

# 将文本转换成mel频谱图
audio_mel_spec,hiddens = process_audio_tokens(
    prompt=text,
    max_length=max_length,
)

# 清理模型
del model, tokenizer, decoder

# 加载Mel转换器
# 尝试再重新生成音频信号
model = Vocos.from_hparams(os.path.join(
    model_dir, 'config/vocos.yaml')).eval()
model.load_state_dict(torch.load(os.path.join(
    model_dir, 'asset/Vocos.pt'), map_location='cpu'))

with torch.no_grad():
    all_wavs = model.decode(torch.cat(audio_mel_spec, dim=2))

# 保存在本地
torchaudio.save("gpt_output.wav", all_wavs, 24000)
