"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional, List, Tuple, Any

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest


def create_test_prompts(lora_path: str) -> list[tuple[str, Any, None] | tuple[str, Any, Any]]:
    """Create a list of test prompts with their sampling parameters.
   ### 这个可以同时传入多个lora adapter，并且同时对多个lora adapter进行inference，比如5个lora adapter，然后一次回复5个结果v##
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None), # 这个是base model 去推理因为none带表没有lora adapter
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        max_tokens=128), None), # 这个是base model 因为none带表没有lora adapter
        (
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        SamplingParams(temperature=0.0,
                       logprobs=1,
                       prompt_logprobs=1,
                       max_tokens=128,
                       stop_token_ids=[32003]), # 这个是lora adapter，把none替换成sql-lora这个lora adapter
        LoRARequest("sql-lora", 1, lora_path)), #
        
        
        ##下面都是一些example，比如lora1 lora2 lora3 lora4
        (
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
        SamplingParams(n=3,
                       best_of=3,
                       use_beam_search=True,
                       temperature=0,
                       max_tokens=128,
                       stop_token_ids=[32003]),
        LoRARequest("sql-lora", 1, lora_path)),
        (
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        SamplingParams(temperature=0.0,
                       logprobs=1,
                       prompt_logprobs=1,
                       max_tokens=128,
                       stop_token_ids=[32003]),
        LoRARequest("sql-lora2", 2, lora_path)),
        (
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
        SamplingParams(n=3,
                       best_of=3,
                       use_beam_search=True,
                       temperature=0,
                       max_tokens=128,
                       stop_token_ids=[32003]),
        LoRARequest("sql-lora", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    results = []
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                #request_output 是个json，里面有很多信息，比如request_id, outputs, status, error？？？可能
                response = request_output.outputs[0].text
                print(response)
                results.append(response)
    return results

def initialize_engine(model_name_or_path, max_loras, max_lora_rank, max_num_seqs, tensor_parallel_size) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model=model_name_or_path,
                             enable_lora=True,
                             max_loras=max_loras,
                             max_lora_rank=max_lora_rank,
                             max_cpu_loras=2,
                             max_num_seqs=max_num_seqs,
                             tensor_parallel_size=tensor_parallel_size)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    model_name_or_path = "meta-llama/Llama-2-7b-hf"
    max_loras = 1
    max_lora_rank = 64
    max_num_seqs = 256
    engine = initialize_engine(model_name_or_path, max_loras, max_lora_rank, max_num_seqs)
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()
