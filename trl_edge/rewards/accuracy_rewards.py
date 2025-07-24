import re
from .math_grader import boxed_reward_fn

def think_accuracy_reward(completions, **kwargs):
    # Regular expression to capture content inside \boxed{}
    completion_contents = [completion[0]["content"] for completion in completions] 
    matches = [re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL) for completion in completion_contents]
    contents = [match.group(1) if match else "" for match in matches]
    #import pdb; pdb.set_trace()
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, kwargs["answer"])]

def box_accuracy_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions] 
    #matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completion_contents]
    #contents = [match.group(1).strip() if match else "" for match in matches]
    result = []
    for c, gt in zip(completion_contents, kwargs["answer"]):
        info, r = boxed_reward_fn(c, gt, fast=False)
        result.append(r)
    return result