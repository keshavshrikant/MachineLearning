import os
import sys
from typing import List


import torch
import transformers
from datasets import load_dataset
from prompt_gen import Prompter

def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result


def generate_and_tokenize_prompt(data_point, train_on_inputs, add_eos_token):
        prompter = Prompter(template_name="gemma", verbose=True)
        full_prompt = prompter.generate_prompt(
            "",
            [data_point["actual_text"], data_point['rewritten_text']],
            data_point["instruction"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                "", [data_point["actual_text"], data_point['rewritten_text']]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt