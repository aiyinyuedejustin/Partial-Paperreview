import requests
import json

# 定义请求的 URL
url = 'http://localhost:8000/v1/chat/completions'

# 定义请求的头部信息
headers = {'Content-Type': 'application/json'}

# 定义请求的参数（即请求体中的 JSON 数据）
data = {"model": "AcademicGPT",
        "max_tokens": 900,
        'top_p': 0.9,
        'temperature': 0.35,
        'repetition_penalty': 1.0,
        'stop_token_ids': [2]
        }

system_message = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
#instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows."
instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria. The given paper is as follows."

with open("inference_output_vllm.txt", "w", encoding="utf-8") as file:
    # 加载数据
    # test_file = "paper_review_data_longqlora_10pct.jsonl"
    test_file = "paper_review_data_longqlora_infer.json"

    with open(test_file, 'r', encoding='utf8') as f:
        data_list = f.readlines()

    example_count = 10

    for data_example in data_list[:example_count]:
        data_example = json.loads(data_example)
        input = data_example["input"].strip()
        data["messages"] = [{"role": "user", "content": input}]

        print(data)

        # 发送 POST 请求
        response = requests.post(url, headers=headers, json=data)

        # 打印响应内容
        print(response.text)