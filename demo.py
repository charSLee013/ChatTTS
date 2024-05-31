import random
import wave
import numpy as np
import torchaudio
import ChatTTS
from scipy.io.wavfile import write

# from zh_normalization import TextNormalizer
import logging
import torch
from IPython.display import Audio

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.DEBUG)

SEED = 1122

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/ChatTTS')

# 加载模型
chat = ChatTTS.Chat()
chat.load_models(
    source="local",
    local_path=model_dir,
    device='cpu'
    # compile=False,
)

torch.manual_seed(SEED) # 音色种子
params_infer_code = {
    'spk_emb': chat.sample_random_speaker(),
    'temperature': 0.00000001,
    'top_P': 0.7,
    'top_K': 20,
}

params_refine_text = {'prompt': '[speed6]'}

texts = ["So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with",
        #  "游戏发生在一个被称作「提瓦特」的幻想世界，在这里，被神选中的人将被授予「神之眼」，导引元素之力。",
        #  "你将扮演一位名为「旅行者」的神秘角色，在自由的旅行中邂逅性格各异、能力独特的同伴们，和他们一起击败强敌，找回失散的亲人——同时，逐步发掘「原神」的真相。",
        #  "因为你的素养很差，我现在每天玩原神都能赚150原石，每个月差不多5000原石的收入， 也就是现实生活中每个月5000美元的收入水平，换算过来最少也30000人民币，",
        #  "虽然我只有14岁，但是已经超越了中国绝大多数人(包括你)的水平，这便是原神给我的骄傲的资本。",
        #  "毫不夸张地说，《原神》是miHoYo迄今为止规模最为宏大，也是最具野心的一部作品。",
        #  "即便在经历了8700个小时的艰苦战斗后，游戏还有许多尚未发现的秘密，错过的武器与装备，以及从未使用过的法术和技能。",
        #  "尽管游戏中的战斗体验和我们之前在烧机系列游戏所见到的没有多大差别，但游戏中各类精心设计的敌人以及Boss战已然将战斗抬高到了一个全新的水平。",
        #  "就和几年前的《 塞尔达传说 》一样，《原神》也是一款能够推动同类游戏向前发展的优秀作品。",
        ]

# 对文本进行预处理
new_texts = []


def filter_punctuation(text):
    allowed_punctuations = {".", ",", "!", "?", "，", "。", "！", "？"," "}
    new_text = ""
    for char in text:
        if char.isalnum() or char in allowed_punctuations:
            new_text += char
    return new_text


# 使用新函数替换原有的预处理步骤
for t in texts:
    filter_text = filter_punctuation(t)
    # 调用模型显示实际生存的文本
    filter_text = chat.infer(
        text=filter_text, skip_refine_text=False, refine_text_only=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code)[0]
    logging.info(f"输入文本: {t}\n预处理后的文本: {filter_text}")
    new_texts.append(filter_text)

torch.manual_seed(SEED) # 推理种子
all_wavs = chat.infer(new_texts, use_decoder=True,
                params_infer_code=params_infer_code,
                skip_refine_text=True,
                params_refine_text=params_refine_text)

# 确保所有数组的维度都是 (1, N)，然后进行合并
combined_wavs = np.concatenate(all_wavs, axis=1)

# audio_file = "./output.wav"
# # 将音频数据缩放到[-1, 1]范围内，这是wav文件的标准范围
# audio_data = combined_wavs / np.max(np.abs(combined_wavs))

# # 将浮点数转为16位整数，这是wav文件常用格式
# audio_data_int16 = (audio_data * 32767).astype(np.int16)

# # 保存到本地，注意采样率为24000
# with wave.open(audio_file, 'wb') as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(24000)
#     wf.writeframes(audio_data_int16)

# Perform inference and play the generated audio
wavs = chat.infer(texts)
Audio(wavs[0], rate=24_000, autoplay=True)


# # Save the generated audio 
torchaudio.save("output.wav", torch.from_numpy(combined_wavs), 24000)