from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import math
from loguru import logger
from component.utils import ModelUtils

"""
使用该脚本，将lora的权重合并到base model中
"""

# 如果想让 base和lora 先分开放，最终inference的时候再合并，才用这个
def load_lora_with_base_model(model_name_or_path, adapter_name_or_path, context_size=12288, load_in_4bit=False):
        # Set RoPE scaling factor
        config = AutoConfig.from_pretrained(model_name_or_path)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        # 加载模型
        model = ModelUtils.load_model(
            model_name_or_path,
            config=config,
            load_in_4bit=load_in_4bit,
            adapter_name_or_path=adapter_name_or_path
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # llama不支持fast
            use_fast=False if model.config.model_type == 'llama' else True
        )

        return model, tokenizer


# 如果想直接合并起来，直接用这个，就剩下一个完整的模型了
def merge_lora_to_base_model(model_name_or_path,
                             adapter_name_or_path,
                             save_path,
                             context_size=12288):
    """"
    model_name_or_path = 'NousResearch/Llama-2-7b-hf'
    adapter_name_or_path = 'LongQLoRA-Llama2-7b-8k-lora'
    save_path = '../checkpoint/llama2-7b-longqlora-8k'
    """

    config = AutoConfig.from_pretrained(model_name_or_path)
    # 修改RoPE的position最大长度
    model_max_length = context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        logger.info(f'Change model_max_length from {orig_ctx_len} to {model_max_length}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    # 更新base model的部分权重
    trainable_params_file = os.path.join(adapter_name_or_path, "trainable_params.bin")
    if os.path.isfile(trainable_params_file):
        model.load_state_dict(torch.load(trainable_params_file, map_location=model.device), strict=False)
    # 合并lora权重
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
