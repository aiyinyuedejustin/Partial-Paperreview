conda activate longqlora
export CUDA_LAUNCH_BLOCKING=1
deepspeed --num_gpus=2 train.py --train_args_file ./train_args/llama2-7b-chat-sft-bf16.yaml
