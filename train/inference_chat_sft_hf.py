import json
from transformers import TextIteratorStreamer
import torch
import math
import time

import sys
sys.path.append("../../")

from component.merge_lora import load_lora_with_base_model

'''
用于inference chat的例子--跟july原版差不多其实， 单纯吧merge lora和base的代码写到了component.merge_lora.py里，一个函数替换掉merge的逻辑
'''

def hf_inference_chat():
    context_size = 12288
    # 使用base model和adapter进行推理
    model_name_or_path = '/root/AcademicGPTLongQLoRA/model_download/Llama-2-7b-chat-hf'
    adapter_name_or_path = './output/LongQLoRA-Llama2-7b-8k/checkpoint-144'

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    model, tokenizer = load_lora_with_base_model(model_name_or_path, adapter_name_or_path,
                                                 context_size=context_size, load_in_4bit=load_in_4bit
    )

    # 生成超参配置
    gen_kwargs = {
        'max_new_tokens': 900, #这是output的max_length
        'top_p': 0.9,
        'temperature': 0.35,
        'repetition_penalty': 1.0,
        'do_sample': True
    }

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)

    gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    gen_kwargs["streamer"] = streamer

    template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:\n")

    instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows.\n\n\n".strip()
    template = template.replace("{{instruction}}", instruction)
    
    with open("inference_output_hf.txt", "w", encoding="utf-8") as file:
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

            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            gen_kwargs["input_ids"] = input_ids

            with torch.no_grad():
                model.generate(**gen_kwargs)

                # print('Output:')
                file.write('Output:\n')

                for new_text in streamer:
                    file.write(new_text)  # 写入文件而不是打印到控制台
                    # print(new_text, end='', flush=True)

                    generated_tokens_count += len(new_text.split())

        print("Avg prompt tokens throughput: {} tokens/s".format(
            round(prompt_tokens_count / (time.time() - start_time), 2))
        )

        print("Avg generated tokens throughput: {} tokens/s".format(
            round(generated_tokens_count / (time.time() - start_time), 2))
        )

        del data_list


if __name__ == '__main__':
    hf_inference_chat()