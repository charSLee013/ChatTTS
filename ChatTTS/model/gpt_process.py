import torch
from .process_types import TaskStatus, FrontendTaskRequest, BackendProcessingTask
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor


class GPTProcessor:

    def __init__(self, max_processing_limit=8, max_watting_limit=64):
        self.max_processing_limit = max_processing_limit
        # 内置处理任务队列,数量跟max_processing_limit一样
        self.processing_queue = [None for _ in range(max_processing_limit)]
        self.request_queue = []  # 前端请求存储列表
        self.max_watting_limit = max_watting_limit

    def submit_task(self, frontend_task: FrontendTaskRequest) -> bool:
        if len(self.request_queue) > self.max_watting_limit:
            return False
        self.request_queue.append(frontend_task)
        return True

    @classmethod
    def create_logits_processor_list(cls, frontend_task: FrontendTaskRequest) -> LogitsProcessorList:
        # 根据frontend_task的参数创建LogitsProcessorList
        return LogitsProcessorList([
            TemperatureLogitsWarper(temperature=frontend_task.temperature),
            TopKLogitsWarper(top_k=frontend_task.top_k, min_tokens_to_keep=3),
            TopPLogitsWarper(top_p=frontend_task.top_p, min_tokens_to_keep=3),
            RepetitionPenaltyLogitsProcessor(
                penalty=frontend_task.repetition_penalty),
        ])

    def clean_completed_or_cancelled_tasks(self):
        self.processing_queue = [task for task in self.processing_queue if task.status not in [
            TaskStatus.COMPLETED, TaskStatus.CANCELLED]]

    def fill_processing_queue(self):
        while True:
            # 1. 检查 processing_queue 是否有空闲的位置
            free_slots = [i for i, task in enumerate(self.processing_queue) if task is None]
            
            # 如果 processing_queue 满了，跳出循环
            if not free_slots:
                break
            
            # 2. 检查 request_queue 是否有请求，如果有请求则按照顺序进行遍历请求
            all_none = True  # 用于检查所有请求的 get_text() 是否都返回 None
            for request in self.request_queue:
                # 3. 从请求的 get_text 函数里面获取具体的 FrontendTaskRequest 任务信息
                frontend_task_request:FrontendTaskRequest = request.get_text()
                if frontend_task_request is not None:
                    all_none = False  # 至少有一个请求的 get_text() 返回了有效值
                    
                    # 4. 将空的编号，请求以及 FrontendTaskRequest 都进行构建成 BackendProcessingTask 推送到指定编号的 processing_queue 中
                    queue_index = free_slots.pop(0)  # 获取一个空闲位置
                    backend_task = BackendProcessingTask(
                        frontend_request=request,
                        queue_index=queue_index,
                        task_request_index=frontend_task_request.task_request_index,
                        input_ids=None,  # 初始化时为 None，后续处理时会填充
                        attention_mask=None,  # 初始化时为 None，后续处理时会填充
                        cache_position=None,  # 初始化时为 None，后续处理时会填充
                        logits_list=GPTProcessor.create_logits_processor_list(frontend_task=frontend_task_request)
                    )
                    self.processing_queue[queue_index] = backend_task
                    
                    # 如果 processing_queue 满了，跳出循环
                    if not free_slots:
                        break
            
            # 如果 request_queue 为空，跳出循环
            if not self.request_queue:
                break
            
            # 如果所有请求的 get_text() 都返回 None，跳出循环
            if all_none:
                break   
    
    def run(self):
        """循环处理.建议单独开一个线程来运行
        """
        while True:
            self.fill_processing_queue()

            """1. 从processing_queue获取数据
            2. 对数据进行预处理，尤其是第一次预处理
            3. 将数据推送到GPT模型进行处理
            4. 将eos_token_id的任务设置为False
            5. 将处理结果推送到streamer中
            """

            self.clean_completed_or_cancelled_tasks()

    @staticmethod
    def clean_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
