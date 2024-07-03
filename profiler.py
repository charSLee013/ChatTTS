import torch
from torch.profiler import profile, record_function, ProfilerActivity

#模型下载
from modelscope import snapshot_download

import ChatTTS
model_dir = snapshot_download('mirror013/ChatTTS')
SEED = 1397

# 加载模型
chat = ChatTTS.Chat()
chat.load_models(
    source="local",
    local_path=model_dir,
    device='mps',
    compile=False,
)

torch.manual_seed(SEED) # 音色种子
# load from local file if exists
params_infer_code = {
    'spk_emb':chat.sample_random_speaker(),
    # "spk_emb":torch.randn(768),
    'temperature': 0.1,
    'top_P': 0.9,
    'top_K': 50,
}

# params_refine_text = {}
params_refine_text = {'prompt': '[oral_6][laugh_2][break_3]'}

# 使用torch.profiler进行性能分析
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("generate"):
        all_wavs = chat.infer(["接下来,杨叔，借我看一下现场地图。"], use_decoder=False,
                params_infer_code=params_infer_code,
                skip_refine_text=False,
                params_refine_text=params_refine_text,
                do_text_normalization=False)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))