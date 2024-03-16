import json
import os.path as osp

from constants import FINETUNE_TEMPLATES

class Prompter(object):

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template = FINETUNE_TEMPLATES['gemma']
        else:
            template = FINETUNE_TEMPLATES[template_name]
        
        self.template = template
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    

    def truncate_txt(self, text, length):
        text_list = text.split()
        
        if len(text_list) <= length:
            return text
        
        return " ".join(text_list[:length])

    
    def generate_prompt(self, instruction, input = None, label = None):
        if input:
            res = self.template["prompt_input"].format(
                original_text=self.truncate_txt(input[0], 200), rewritten_text=self.truncate_txt(input[1], 200)
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
    

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

