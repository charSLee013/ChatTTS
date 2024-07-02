
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
from ..utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat

def infer_code(
    models,
    text, 
    spk_emb = None,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.3, 
    repetition_penalty = 1.05,
    max_new_token = 2048,
    **kwargs
):
    
    device = next(models['gpt'].parameters()).device
    
    if not isinstance(text, list): 
        text = [text]
        
    if not isinstance(temperature, list):
        temperature = [temperature] * models['gpt'].num_vq
    
    if spk_emb is not None:
        text = [f'[Stts][spk_emb]{i}[Ptts]' for i in text] 
    else:
        text = [f'[Stts][empty_spk]{i}[Ptts]' for i in text]
    
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)
    input_ids = text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq)
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)
    
    inputs = {
        'input_ids': input_ids,
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }

    emb = models['gpt'].get_emb(**inputs)
    if spk_emb is not None:
        # 获取 [spk_emb] 标记的 token ID
        spk_emb_token_id = models['tokenizer'].convert_tokens_to_ids('[spk_emb]')
        
        # 找到 input_ids 中等于 spk_emb_token_id 的位置
        spk_emb_positions = inputs['input_ids'][..., 0] == spk_emb_token_id
        
        # 这部分代码将 spk_emb 移动到与模型相同的设备(CPU或GPU)。
        # device 是通过 next(models['gpt'].parameters()).device 获取的。
        spk_emb_device = spk_emb.to(device)

        # 这部分代码将 spk_emb 转换为与 emb 相同的数据类型
        # emb.dtype 是嵌入表示的类型，通常是 torch.float32 或 torch.float16。
        spk_emb_dtype = spk_emb_device.to(emb.dtype)

        # 这部分代码在 spk_emb 的第一个维度上增加一个新的维度，使其可以进行扩展操作。
        # 增加的维度使 spk_emb 从形状 [embedding_dim] 变为 [1, embedding_dim]。
        spk_emb_expanded = spk_emb_dtype[None]

        # 这部分代码将 spk_emb 的第一个维度扩展到与文本数量相同。
        # len(text) 是输入文本的数量，-1 表示保持第二个维度不变。扩展后的形状为 [len(text), embedding_dim]
        spk_emb_expanded = spk_emb_expanded.expand(len(text), -1)

        # 归一化的结果是每个嵌入向量的 L2 范数为 1。
        normalized_spk_emb = F.normalize(
            spk_emb_expanded, 
            p=2.0,  # 使用 L2 范数(欧几里得范数)进行归一化。
            dim=1,  # 在第一个维度上进行归一化，即对每个嵌入向量进行归一化。
            eps=1e-12 # 一个小的 epsilon 值，用于避免除零错误。 
        )
        
        # 将 emb 中对应 spk_emb_positions 的位置替换为 normalized_spk_emb
        emb[spk_emb_positions] = normalized_spk_emb
        
        # emb[inputs['input_ids'][..., 0] == models['tokenizer'].convert_tokens_to_ids('[spk_emb]')] = \
        #     F.normalize(spk_emb.to(device).to(emb.dtype)[None].expand(len(text), -1), p=2.0, dim=1, eps=1e-12)
    
    num_code = models['gpt'].emb_code[0].num_embeddings - 1
    
    LogitsWarpers = []
    if top_P is not None:
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(\
            repetition_penalty, num_code, 16))
    
    result = models['gpt'].generate(
        emb, inputs['input_ids'], 
        temperature = torch.tensor(temperature, device=device), 
        attention_mask = inputs['attention_mask'],
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = num_code, 
        max_new_token = max_new_token, 
        infer_text = False,
        **kwargs
    )
    
    return result


def refine_text(
    models, 
    text,
    top_P = 0.7, 
    top_K = 20, 
    temperature = 0.7, 
    repetition_penalty = 1.2,
    max_new_token = 384,
    prompt = '',
    **kwargs
):
    # 确保text参数是一个列表
    device = next(models['gpt'].parameters()).device
    if not isinstance(text, list): 
        text = [text]
    assert len(text), 'text should not be empty'
    
    # 给每个输入文本前加上特殊标记[Sbreak]，用于模型识别不同的文本开始
    # 然后加上[Pbreak]和prompt，prompt是需要合成的语音的文本
    text = [f"[Sbreak]{i}[Pbreak]{prompt}" for i in text]
    
    # 模型的tokenizer将文本转换为token IDs，并返回PyTorch张量
    text_token = models['tokenizer'](text, return_tensors='pt', add_special_tokens=False, padding=True).to(device)

    """
    这里的展开讲讲text_token里面的内容
    包括三样东西: input_ids, text_mask, attention_mask
    # input_ids: 输入文本的token IDs，形状为(batch_size, sequence_length)
    它包含了原始文本被转换成的token的ID序列。每一个整数ID实际上对应着tokenizer词典中的一个特定的token。
    例如，如果词汇表中“hello”这个词的ID是5，那么在文本中出现“hello”的地方就会被编码为数字5。input_ids是模型进行预测的主要输入。

    # token_type_ids: 一个布尔张量，表示每个token是否是文本的一部分，主要用于区分输入文本中不同句子或段落的来源。在处理像BERT这样的模型时，这尤其重要，因为这些模型需要知道输入的哪部分来自第一个句子，哪部分来自第二个句子(比如在问答或自然语言推理任务中)。
    token_type_ids通常在处理两个句子作为输入的情况下使用，其中第一个句子的所有token可能被标记为0，而第二个句子的所有token被标记为1。如果输入只有一个句子，那么整个token_type_ids序列可能都为0。


    # attention_mask: 一个布尔张量，它用于指示哪些位置的token是实际的文本内容，哪些位置是由于填充(padding)而添加的。在处理不同长度的文本序列时，为了能够批量处理，通常会将较短的序列填充到相同的长度。attention_mask中的1表示对应的input_ids位置是有效的文本内容，而0则表示该位置是填充的，模型应该忽略这部分。这对于自注意力机制来说至关重要，因为它告诉模型在计算注意力权重时哪些位置应该被考虑，哪些应该被忽略
    """
    
    # 创建一个与输入形状相同的文本掩码，所有值为True。
    text_mask = torch.ones(text_token['input_ids'].shape, dtype=bool, device=device)

    """
    # 创建一个字典，包含输入的ID张量、文本掩码和注意力掩码，扩展维度以适应模型的预期输入格式
    # text_token['input_ids'] 的形状为 (batch_size, sequence_length)
    # [...,None] 扩展成 (batch_size,sequence_length,1)
    # expand(-1, -1, models['gpt'].num_vq) 保留前面的两个维度不变，只改变第三个维度为 models['gpt'].num_vq
    # 最后形状变成 (batch_size, sequence_length, num_vq)

    继续展开说说为啥要调整形状，这一部分需要联系GPT_warpper来了解
    在GPT_warpper模型设计中，num_vq代表了多个嵌入层的数量，每个嵌入层负责将不同的音频或潜在特征编码为模型的隐藏维度。因此，原始的文本输入(text_token['input_ids'])需要扩展到适合这些嵌入层的形状，以便模型能够并行处理文本和潜在的多维度特征。
    这一操作的目的是将 input_ids 扩展为包含多个向量量化(vector quantization, VQ)版本的输入数据。具体来说：
    向量量化(VQ)：在某些模型中，特别是涉及多模态(如文本和音频)的模型中，可能会有多个向量量化层，每个层将输入数据映射到不同的嵌入空间。
    扩展维度：通过将 input_ids 扩展到 (batch_size, sequence_length, models['gpt'].num_vq)，可以为每个 token ID 生成多个向量量化版本的嵌入表示。
    """
    inputs = {
        'input_ids': text_token['input_ids'][...,None].expand(-1, -1, models['gpt'].num_vq),
        'text_mask': text_mask,
        'attention_mask': text_token['attention_mask'],
    }
    # 实现top-K或者top-P采样
    LogitsWarpers = []
    if top_P is not None:
        # 词汇累积概率
        LogitsWarpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        # 保留的最高概率词汇的数量
        LogitsWarpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))
        
    LogitsProcessors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        # 对已经生成的词汇施加惩罚，减少它们再次被选中的概率。
        LogitsProcessors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, len(models['tokenizer']), 16))
    
    result = models['gpt'].generate(
        models['gpt'].get_emb(**inputs), # 输入的嵌入表示，通常是通过模型的嵌入层生成的。
        inputs['input_ids'],    # 输入文本的token IDs。
        temperature = torch.tensor([temperature,], # 控制生成文本多样性的参数。较高的温度会使生成的文本更加多样化，较低的温度会使生成的文本更加确定性。
                                   device=device), 
        attention_mask = inputs['attention_mask'], # 这是一个掩码，用于指示哪些token是有效的，哪些是填充的。
        LogitsWarpers = LogitsWarpers,
        LogitsProcessors = LogitsProcessors,
        eos_token = torch.tensor(models['tokenizer'].convert_tokens_to_ids('[Ebreak]'), device=device)[None], # 这是结束标记的token ID，当生成的文本包含这个token时，生成过程会停止。
        max_new_token = max_new_token,  # 这是生成的最大token数量。
        infer_text = True,
        **kwargs
    )
    return result