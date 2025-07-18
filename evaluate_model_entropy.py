# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import time

import fire
import numpy as np
import torch
import vllm
from jinja2 import Template
# 新增 transformers 的 AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk
from understand_r1_zero.math_grader import (answer_tag_reward_fn,
                                            boxed_reward_fn)


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


# The following two templates are used to evaluate baselines from other projects.
def apply_prime_zero_template(question: str):
    """https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero"""
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"


def apply_open_reasoner_zero_template(question: str):
    "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/playground/zero_setting_base.py"
    prompt_template_jinja = """\
    {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
    The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
    Assistant: <think>\
    """
    prompt_instruction_template_jinja = """\
    You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
    This is the problem:
    {{prompt}}
    """
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    template: str = "qwen_math",
    dataset_name: str = "/data/vlm/zxj/understand-r1-zero/datasets/evaluation_suite",
    temperature: float = 0.1,
    top_p: float = 1,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    max_test: int = 999999,
    save: bool = False,
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=int(time.time_ns()),
    )

    model = vllm.LLM(
        model_name,
        swap_space=32,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    scoring_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config.torch_dtype
    ).to(device)
    scoring_model.eval()
    

    if "prime" in model_name.lower():
        template = "prime-zero"
    if "open-reasoner-zero" in model_name.lower():
        template = "open-reasoner-zero"

    if "instruct" in model_name.lower() and "instruct" not in template:
        input(
            f"{model_name}\n{template}\ninstruct model but not instruct template! continue?"
        )

    print("Using template:", template)
    if template in ["qwen_math", "no"]:
        math_reward_fn = boxed_reward_fn
        if template == "qwen_math":
            apply_template = apply_qwen_math_template
        else:
            apply_template = lambda x: x

    elif template == "r1":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template
    elif template == "prime-zero":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_prime_zero_template
    elif template == "open-reasoner-zero":
        from understand_r1_zero.math_grader import answer_tag_reward_fn_for_orz

        math_reward_fn = answer_tag_reward_fn_for_orz
        apply_template = apply_open_reasoner_zero_template
    elif template == "llama-instruct":
        math_reward_fn = boxed_reward_fn
        # Tokenizer already loaded at the top
        def apply_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d":  # r1-distill
        math_reward_fn = boxed_reward_fn

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        raise ValueError

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    to_be_saved = []
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue
        prompts = dataset["problem"][:max_test]
        targets = dataset["answer"][:max_test]
        prompts = list(map(apply_template, prompts))
        print("Inference for ", task_name)
        # --- 步骤 1: 使用 vLLM 生成 ---
        outputs = model.generate(prompts, sampling_params)
        
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        
        for k in range(len(outputs)):
            output = outputs[k]
            prompt_text = output.prompt
            gt_repeated = [targets[k]] * sampling_params.n
            
            rewards, infos = [], []
            model_outputs_text = [o.text for o in output.outputs]
            entropies = []
            all_max_entropy_positions, all_max_entropy_texts = [], []
            all_min_entropy_positions, all_min_entropy_texts = [], []

            for i, sample_output in enumerate(output.outputs):
                gt = targets[k]
                info, r = math_reward_fn(sample_output.text, gt, fast=False)
                rewards.append(r)
                infos.append(info)
                
                prompt_len = len(output.prompt_token_ids)
                full_token_ids = torch.tensor([output.prompt_token_ids + sample_output.token_ids], device=device)
                
                token_entropies = []
                with torch.no_grad():
                    logits = scoring_model(input_ids=full_token_ids).logits
                    print("logits:",logits.shape)

                    gen_logits = logits[0, prompt_len - 1 : -1, :]
                    gen_logits = gen_logits / temperature #torch.Size([seq_len, 151936])
                    print("gen_logits:",gen_logits.shape)
                    if gen_logits.shape[0] > 0: 
                        log_probs = torch.log_softmax(gen_logits, dim=-1)
                        probs = torch.softmax(gen_logits, dim=-1)
                        print("log_probs:",log_probs.shape)
                        print("probs:",probs.shape)

                        token_entropies_tensor = -(probs * log_probs).sum(dim=-1)
                        print("token_entropies_tensor:",token_entropies_tensor.shape)
                        token_entropies = token_entropies_tensor.cpu().tolist() #torch.Size([seq_len])

                avg_entropy = 0.0
                max_entropy_positions, max_entropy_texts = [], []
                min_entropy_positions, min_entropy_texts = [], []

                if token_entropies:
                    avg_entropy = np.mean(token_entropies)
                    token_entropies_tensor = torch.tensor(token_entropies)
                    k_val = min(30, len(token_entropies))

                    if k_val > 0:
                        max_indices = token_entropies_tensor.topk(k=k_val).indices
                        max_entropy_positions = max_indices.tolist()
                        max_entropy_texts = [tokenizer.decode(sample_output.token_ids[pos]) for pos in max_entropy_positions]
                        #max_entropy_texts = tokenizer.convert_ids_to_tokens(max_tokens)
                        #max_entropy_texts = [token.replace('\u0120', ' ') if token is not None else '' for token in max_entropy_texts]
                        
                        sorted_indices = torch.argsort(token_entropies_tensor)
                        min_indices_list = []
                        for idx in sorted_indices:
                            if token_entropies_tensor[idx] > 1e-6: 
                                min_indices_list.append(idx.item())
                            if len(min_indices_list) == k_val:
                                break
                        min_entropy_positions = min_indices_list
                        min_entropy_texts = [tokenizer.decode(sample_output.token_ids[pos]) for pos in min_entropy_positions]
                        #min_entropy_texts = tokenizer.convert_ids_to_tokens(min_tokens)
                        #min_entropy_texts = [token.replace('\u0120', ' ') if token is not None else '' for token in min_entropy_texts]

                
                entropies.append(avg_entropy)
                all_max_entropy_positions.append(max_entropy_positions)
                all_max_entropy_texts.append(max_entropy_texts)
                all_min_entropy_positions.append(min_entropy_positions)
                all_min_entropy_texts.append(min_entropy_texts)
            
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            if infos and infos[0] is not {}:
                valid_infos = [i["formatted"] for i in infos if i and "formatted" in i]
                if valid_infos:
                    batch_formatted.append(np.array(valid_infos).sum())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": prompt_text,
                    "gt": gt_repeated,
                    "model_output": model_outputs_text,
                    "reward": [r for r in rewards],
                    "entropy": entropies,
                    "max_entropy_positions": all_max_entropy_positions,
                    "max_entropy_texts": all_max_entropy_texts,
                    "min_entropy_positions": all_min_entropy_positions,
                    "min_entropy_texts": all_min_entropy_texts,
                }
            )

        results[task_name] = np.mean(batch_scores) if batch_scores else 0
        avg_lens[task_name] = np.mean(batch_lengths) if batch_lengths else 0
        if batch_formatted:
            formatted[task_name] = np.mean(batch_formatted)
        max_lens[task_name] = np.max(batch_lengths) if batch_lengths else 0

    print(results)
    print("avg:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    print("formatted:", formatted)

    model_name_str = model_name.split("/")[-1]
    tem = {0.1: "01", 0.3: "03", 0.6: "06", 1: "1", 1.0: "1", 1.2: "12"}
    fn = f"/data/vlm/zxj/understand-r1-zero/result/vllm-t{tem[temperature]}/{model_name_str}.json"
    print(f"saving model outputs at {fn}")
    json.dump(
        to_be_saved,
        open(fn,"w",),
        indent=4,
    )


if __name__ == "__main__":
    fire.Fire(main)