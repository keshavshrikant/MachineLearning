import pandas as pd
import yaml
import json
import re
import os
import random
import tqdm

from databricks_dolly import Dolly
from constants import GENERATE_TRANSFORMATION_SYS_MSG
from genai_client import generate_text
from random import shuffle
from openai import OpenAI

from local_settings.Settings import OPENAI_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

class TransformParagraph:

    def __init__(self, dolly_path, seed_tasks_path, instruction_dir):
        self.dolly_path = dolly_path
        self.dolly = Dolly(self.dolly_path)
        self.data = self.dolly.read_databricks_dolly_data()
        self.seed_tasks_path = seed_tasks_path
        self.instruction_dir = instruction_dir
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

    
    def create_prompt(self, instruction, input_para):
        prompt = GENERATE_TRANSFORMATION_SYS_MSG.format(**{"para": input_para, "instruction": instruction})
        return prompt


    def post_process_gpt_response(self, response):
        out_json = response[response.find("{"): response.rfind("}")+1]        
        try:
            out_json = yaml.load(out_json, yaml.SafeLoader)
            return out_json["Rewritten Paragraph"]
        except:
            match = re.findall(r"rewritten paragraph(.*?)}", response, re.I)
            if match:
                return match[0].replace("\"", "")
        return None

    
    def generate(self):
        # Load Paragraphs
        self.creative_data = self.data[self.data['category'] == "creative_writing"]
        paragraphs = self.creative_data['response'].tolist()        

        # Load Instructions
        seed_instructions = pd.read_csv(self.seed_tasks_path)['instruction'].tolist()        
        print(f"Loaded {len(seed_instructions)} seed-generated instructions")
        
        machine_gen_instructions = []
        if os.path.exists(os.path.join(self.instruction_dir, "machine_generated_instructions.jsonl")):
            with open(os.path.join(self.instruction_dir, "machine_generated_instructions.jsonl"), "r") as fin:
                for line in fin:
                    instruction_info = json.loads(line)
                    machine_gen_instructions.append(instruction_info["instruction"])                    
        
        print(f"Loaded {len(machine_gen_instructions)} machine-generated instructions")

        # Mix and Match paras and instructions
        instructions = seed_instructions + machine_gen_instructions
        shuffle(instructions)
        shuffle(paragraphs)
        
        # Track Progress
        progress_bar = tqdm.tqdm(total=1000)
        if instructions:
            progress_bar.update(0)

        # Transform
        with open(os.path.join(self.instruction_dir, "gpt_generated_tranformations.jsonl"), "a") as fout:
            for _ in range(1000):
                instr = random.choice(instructions)
                para = random.choice(paragraphs)
                prompt = self.create_prompt(instr, para)
                results = generate_text(self.openai_client, prompt, 
                        max_tokens=500,
                        temperature=0.7,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,                    
                        n=1,
                    )
                
                response = self.post_process_gpt_response(results)
                if response is None:
                    continue
                else:
                    fout.write(json.dumps({
                        "instruction": instr,
                        "actual_text": para,                    
                        "rewritten_text": response
                    }) + "\n")
                progress_bar.update(1)



if __name__ == "__main__":
    transform = TransformParagraph(dolly_path="/Users/nisha/Keshav/Kaggle/databricks-dolly-15k.jsonl", 
                                  seed_tasks_path="/Users/nisha/Documents/GitHub/MachineLearning/self_instruct/seed_instructions.csv", 
                                  instruction_dir=".")
    
    transform.generate()
