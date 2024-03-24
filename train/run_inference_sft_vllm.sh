conda activate longqlora_vllm

python -u -m vllm.entrypoints.openai.api_server \
	      --host 0.0.0.0 \
	      --served-model-name AcademicGPT \
	      --model "/root/AcademicGPTLongQLoRA/model_download/Llama-2-7b-chat-hf" \
	      --chat-template="chat_template/template_alpaca_llama2_v2.jinja" \
	      --enable-lora \
	      --max-lora-rank=64 \
	      --max-num-seqs 12288 \
        --lora-modules academic="./output/LongQLoRA-Llama2-7b-8k/checkpoint-144" | tee ~/openai_api_server.log


