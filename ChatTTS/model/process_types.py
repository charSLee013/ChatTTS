from dataclasses import dataclass
from enum import Enum
from abc import ABC,abstractmethod
from typing import List, Optional
from transformers.generation.logits_process import LogitsProcessorList

import torch

from stream import BaseStreamer

class TaskStatus(Enum):
    WAITING = '等待中'
    PROCESSING = '处理中'
    COMPLETED = '已完成'
    CANCELLED = '取消'
    FAILED = '失败' # 重大事故

class TaskType(Enum):
    TEXT = '重写文本'
    AUDIO = 'GPT语音'
    # ALL = '重写并语音' #TODO: 涉及到循环内嵌以后再说

@dataclass
class FrontendTaskRequest():
    text: str
    task_request_index: int

@dataclass
class FrontendTaskProcess():
    id: int
    top_p: float = 0.7
    top_k: int = 50
    temperature: float = 0.3
    repetition_penalty: float =1.05
    task_type: TaskType = TaskType.AUDIO
    stream: BaseStreamer = None
    
    texts: List[str] = None  # 存储子字符串的列表
    current_index: int = 0  # 当前子字符串的索引

    def get_text(self) -> Optional[FrontendTaskRequest]:
        """从前端请求处获取需要预测的文本
        """
        if self.current_index < len(self.texts):
            text = self.texts[self.current_index]
            task_request_index = self.current_index
            self.current_index += 1
            return FrontendTaskRequest(text=text, task_request_index=task_request_index)
        else:
            return None


@dataclass
class BackendProcessingTask:
    frontend_request: FrontendTaskProcess
    queue_index: int        # GPT处理队列的索引，用于标识当前任务在队列中的位置
    task_request_index: int         # 请求里面的任务索引，每个请求可以有多个任务，用来标识当前任务在请求中的位置
    input_ids: torch.tensor = None
    attention_mask: torch.tensor = None
    cache_position: torch.tensor = None
    logits_list: Optional[LogitsProcessorList] = None

@dataclass
class FrontendTaskResponse:
    ids: torch.tensor
    hidden: torch.tensor
    task_request_index: int # 对于自身同属的请求里面的位置索引

