# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from .math_grader import boxed_reward_fn

def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]  # ! change
    # completion_contents = [completion for completion in completions]
    # import pdb; pdb.set_trace()
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def box_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    #pattern = r"\\boxed\{(.*?)\}"
    #matches = [re.search(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    completion_contents = [completion[0]["content"] for completion in completions]
    result = []
    for c, gt in zip(completion_contents, kwargs["answer"]):
        info, r = boxed_reward_fn(c, gt, fast=False)
        result.append(1.0) if info else result.append(0.0)
    return result


