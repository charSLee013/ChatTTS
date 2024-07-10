#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 方便查看源代码
from transformers import (
    EncoderDecoderModel,
    GPT2LMHeadModel,
    GPT2Model,
)
GPT2Model.forward
GPT2LMHeadModel.forward
EncoderDecoderModel.forward
EncoderDecoderModel.generate
GPT2LMHeadModel.prepare_inputs_for_generation


# In[2]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# 示例使用
model_name = "gpt2"  # 使用较小的模型作为示例
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[3]:


input_ids = tokenizer.encode('Your prompt text here', return_tensors='pt')

# 设置top_k和top_p的值
top_k = 50
top_p = 0.9

# 使用model.generate函数生成文本
generated_ids = model.generate(input_ids, 
                               do_sample=True, 
                               max_length=100, 
                               top_k=top_k, 
                               top_p=top_p)

# 解码生成的文本
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)

