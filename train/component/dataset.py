from loguru import logger
import json
from torch.utils.data import Dataset
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        text = json.loads(data)['text']
        return text


class EvalDataset(Dataset):
    """
    用于评测ppl
    """
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100, sliding_window=256):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        token_list = np.memmap(file, dtype=np.uint16, mode='r').tolist()

        # 以滑动窗口截取评测数据
        eval_data_list = []
        for i in range(0, len(token_list), sliding_window):
            input_ids = token_list[i: i+max_seq_length]
            labels = token_list[i: i+max_seq_length]
            # padding
            padding_len = self.max_seq_length - len(input_ids)
            input_ids += [self.pad_token_id]*padding_len
            labels += [self.ignore_index]*padding_len
            eval_data_list.append({
                'input_ids': input_ids,
                'labels': labels
            })
        logger.info("there are {} data in eval dataset".format(len(eval_data_list)))
        self.data_list = eval_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data

# 以下是vicuna的格式原始的， 不用了
class VicunaSFTDataset1111(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        self.input_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        沿袭Vicuna的的格式。
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        USER: xxx
        ASSISTANT: xxx
        """
        data = self.data_list[index]
        data = json.loads(data)
        inputs = data['input'].strip()
        output = data['output'].strip()
        # 输入部分
        input_format = self.input_template.format(input=inputs)
        # 用tokenizer将格式化后的输入文本（即用户问题和预设的模板）转换为模型能够理解的一系列标记ID。
        input_format_ids = self.tokenizer(input_format, add_special_tokens=False).input_ids
        # 将输出文本（即AI助手的回答）转换为标记ID序列，并在序列末尾添加一个结束标记（eos_token_id），表示回答的结束。
        output_ids = self.tokenizer(output, add_special_tokens=False).input_ids + [self.eos_token_id]

        input_ids = input_format_ids + output_ids
        labels = [self.ignore_index] * len(input_format_ids) + output_ids
        assert len(input_ids) == len(labels)

        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        # padding
        padding_len = self.max_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * padding_len
        labels += [self.ignore_index] * padding_len
        attention_mask += [0] * padding_len

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs



#因为train.py用的名字还是这个，保持不变。但是给llama的
# class VicunaSFTDataset(Dataset):

#     def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
#         self.tokenizer = tokenizer
#         self.ignore_index = ignore_index
#         self.max_seq_length = max_seq_length
#         self.pad_token_id = tokenizer.pad_token_id
#         self.eos_token_id = tokenizer.eos_token_id
#         logger.info('Loading data: {}'.format(file))
#         with open(file, 'r', encoding='utf8') as f:
#             data_list = f.readlines()

#         logger.info("there are {} data in dataset".format(len(data_list)))
#         self.data_list = data_list
#         # self.input_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "
#         self.input_template = "You are a professional machine learning conference reviewer who reviews a\
#       given paper and considers 4 criteria: **importance and novelty**, **potential reasons\
#           for acceptance**, **potential reasons for rejection**, and **suggestions for \
#             improvement**.\nThe given paper is as follows using section lables of [TITLE], [ABSTRACT], [CAPTIONS], [CONTENT] to separate paper contents:\n{input}\n"

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         """
#         自定义 input template，并且要 提前对input和output截断，保证output被完整保留！！
#         """
#         data = self.data_list[index]
#         data = json.loads(data)
#         inputs = data['input'].strip()
#         output = data['output'].strip()


#         #先处理output,计算output长度，剩余长度
#         output_ids = self.tokenizer(output, add_special_tokens=False).input_ids + [self.eos_token_id]
#         output_length = len(output_ids)
#         available_input_length = self.max_seq_length - output_length  # 可用的input长度

#         # 输入部分, 拼接 instruction+input
#         input_formated =  self.input_template.format(input=inputs)
#         input_formated_ids = self.tokenizer(input_formated,  add_special_tokens=False).input_ids

#         len_input_formated_ids = len(input_formated_ids)

#         if len_input_formated_ids > available_input_length: #只有input长度大于可用长度才截断
#             input_formated_ids = input_formated_ids[:available_input_length]  # 截断input

#         input_ids = input_formated_ids + output_ids



#         labels = [self.ignore_index] * len(input_formated_ids) + output_ids
#         assert len(input_ids) == len(labels), "Input IDs and labels must be the same length."

#         # 如果input_ids长度大于max_seq_length，那么就截断
#         if len(input_ids) > self.max_seq_length:
#             input_ids = input_ids[:self.max_seq_length]
#             labels = labels[:self.max_seq_length]


#         attention_mask = [1] * len(input_ids)

#         # padding if needed
#         padding_len =  max(0, self.max_seq_length - len(input_ids))
#         input_ids += [self.pad_token_id] * padding_len
#         labels += [self.ignore_index] * padding_len
#         attention_mask += [0] * padding_len

#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels
#         }
#         return inputs
    
## debug后用这个
class Llama2SFTDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.unk_token_id
        self.eos_token_id = tokenizer.eos_token_id
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = [da for (di, da) in enumerate(data_list) if di not in [4356]]
        # self.data_list = data_list
        logger.info("there are {} data in dataset".format(len(self.data_list)))
        self.input_template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                                "Write a response that appropriately completes the request.\n\n"
                                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        llama2官方instruction数据格式用例
        https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/datasets/alpaca_dataset.py#L14

        bos + query(instruction+input) + resp(output) + eos
        """
        # logger.info(f"当前的样本是第{index}个")

        data = self.data_list[index]
        data = json.loads(data)
        inputs = data['input'].strip()
        output = data['output'].strip()

        # 输入部分
        instruction = "You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The given paper is as follows.\n\n\n"
        instruction = instruction.strip()
        # 拼接 instruction+input
        input_format = self.input_template.format(instruction=instruction, input=inputs)

        # source_ids
        input_format_ids = self.tokenizer.encode(input_format, add_special_tokens=False)
        # target_ids
        output_ids = self.tokenizer.encode(output + self.tokenizer.eos_token, add_special_tokens=False)

        # 分别计算输入、输出合适的截断
        max_output_len = int(self.max_seq_length * (len(output_ids) / (len(input_format_ids) + len(output_ids))))
        max_output_len = max(max_output_len, 1)     # 至少保留1个token的output
        max_input_len = self.max_seq_length - max_output_len

        # 对输入、输出进行截断
        if len(input_format_ids) > max_input_len:
            input_format_ids = input_format_ids[:max_input_len]
        if len(output_ids) > max_output_len:
            output_ids = output_ids[:max_output_len]
        
        # mask
        input_format_mask = [self.ignore_index] * len(input_format_ids)

        # concat inputs
        input_ids = input_format_ids + output_ids
        labels = input_format_mask + output_ids # 000000 1111 

        # 再次检查截断
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        
        attention_mask = [1] * len(input_ids)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ", add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
