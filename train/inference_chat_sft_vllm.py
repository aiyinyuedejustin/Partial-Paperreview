import json
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer
import torch
import math
from threading import Thread
import time

import sys
sys.path.append("../../")

from component.multilora_inference import initialize_engine,process_requests
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

'''
用于inference chat的例子---最终版，用vllm的engine
'''

def vLLM_inference_chat():
    # 使用base model和adapter进行推理，分开加载的
    model_name_or_path = '/root/AcademicGPTLongQLoRA/model_download/Llama-2-7b-chat-hf'
    adapter_name_or_path = './output/LongQLoRA-Llama2-7b-8k/checkpoint-144'

    # 不支持量化
    max_loras = 1 # 控制可以在同一批次中使用的LoRAs的数量 ， 比如1个lora adapter就是1，因为我们只有一个lora adapter
    max_lora_rank = 64 # 因为可以加载多个lora，所以这个rank是所有lora的最大rank
    max_num_seqs = 12288 # max_position_embeddings, 最大输入长度
    n_gpu = 2
    
    # create engine的时候，只需要导入base model的名字，不需要导入adapter的名字，后面lora可以加载多个
    #具体initialize_engine去看multi_lora_inference.py
    engine = initialize_engine(model_name_or_path, max_loras, max_lora_rank, max_num_seqs, n_gpu)
    print("成功加载engine!")


    #template要跟微调的时候的template一样，比如system，
    template = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:\n")
    instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows\n\n\n".strip()
    template = template.replace("{{instruction}}", instruction)

    # 生成超参配置
    gen_kwargs = {
        'max_tokens': 900, #这是output的max_length!1111
        'top_p': 0.9,
        'temperature': 0.35,
        'repetition_penalty': 1.0,
        'stop_token_ids': [2]  #eos_token_id?????
    }


#要推理的文件
    with open("inference_output_vllm.txt", "w", encoding="utf-8") as file:
        # 加载数据
        test_file = "paper_review_data_longqlora_10pct.jsonl"
        # test_file = "paper_review_data_longqlora_infer.json"

        with open(test_file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        start_time = time.time()
        example_count = 10
        prompt_tokens_count = 0
        generated_tokens_count = 0

        for data_example in data_list[:example_count]:
            data_example = json.loads(data_example)
            ip = data_example["input"].strip()

            text = ip.strip()
            prompt = template.replace("{{input}}", text)

            prompt_tokens_count += len(prompt.split())

            #传入的参数，是一个dict，所以要用**gen_kwargs
            sampling_params = SamplingParams(**gen_kwargs)
            
            #从这里才开始加载lora adapter， 并且起个名字，明确是哪个adapter，因为可以加载多个，1是index
            lora_request = LoRARequest("academic-lora", 1, adapter_name_or_path)
            test_prompts = [(prompt, sampling_params, lora_request)] # 【prompt, 超参, lora权重】，可以多个tuple，就同时传入多个不同的 (prompt+参数+adapter )，我们这里只有1个adapter

            request_outputs = process_requests(engine, test_prompts) # 相当于合并了add_request和step两个函数，去看multi_lora_inference.py
            for request_output in request_outputs:
                file.write('Output:\n')
                file.write(request_output)

                generated_tokens_count += len(request_output.split())

        print("Avg prompt tokens throughput: {} tokens/s".format(
            round(prompt_tokens_count / (time.time() - start_time), 2))
        )

        print("Avg generated tokens throughput: {} tokens/s".format(
            round(generated_tokens_count / (time.time() - start_time), 2))
        )

        del data_list


if __name__ == '__main__':
    vLLM_inference_chat()
