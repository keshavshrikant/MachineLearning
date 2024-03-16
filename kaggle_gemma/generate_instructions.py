import pandas as pd
import numpy as np
import random
import json
import os

from kaggle_gemma.databricks_dolly import Intructions
from constants import GENERATE_INTRUCTIONS_SYS_MSG
from genai_client import generate_text

from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from openai import OpenAI

from local_settings.Settings import OPENAI_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

class GenerateInstructions:

    def __init__(self, seed_tasks_path, machine_gen_instructions_path, instruction_dir):
        self.seed_tasks_path = seed_tasks_path
        self.machine_gen_instructions_path = machine_gen_instructions_path
        self.instruction_dir = instruction_dir
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))


    def create_prompt(self, instructions):
        prompt = GENERATE_INTRUCTIONS_SYS_MSG + "\n\n"   
        for idx, instr in enumerate(instructions):
            prompt += f"TASK {idx + 1}: {instructions[idx]}\n"

        # prompt += f"TASK {idx + 2}: "
        return prompt
    

    def post_process_gpt_response(self, response):
        instructions = []
        responses = response.split("\n")
        for resp in responses:
            instructions.append(resp[resp.find(":")+1: ].strip())
        
        return instructions

    
    def generate(self):
        
        seed_instructions = pd.read_csv(self.seed_tasks_path)['instruction'].tolist()
        request_idx = 0
        
        machine_gen_instructions = []
        if os.path.exists(os.path.join(self.instruction_dir, "machine_generated_instructions.jsonl")):
            with open(os.path.join(self.instruction_dir, "machine_generated_instructions.jsonl"), "r") as fin:
                for line in fin:
                    instruction_info = json.loads(line)
                    machine_gen_instructions.append(instruction_info["instruction"])
                    request_idx = instruction_info["request_idx"] + 1
        
        print(f"Loaded {len(machine_gen_instructions)} machine-generated instructions")

        prompt_instructions = []
        if machine_gen_instructions == []:
            prompt_instructions = random.choices(seed_instructions, k=8)
        else:
            prompt_instructions = random.choices(seed_instructions, k=6)
            prompt_instructions.extend(random.choices(machine_gen_instructions, k=2))
        
        prompt = self.create_prompt(prompt_instructions)
        # print(prompt)
        results = generate_text(self.openai_client, prompt, 
                    max_tokens=2000,
                    temperature=0.7,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=2,
                    stop_sequences=["\n\n", "\n16", "16.", "16 ."],                    
                    n=1,
                )

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        
        instructions = self.post_process_gpt_response(results)

        with open(os.path.join(self.instruction_dir, "machine_generated_instructions.jsonl"), "a") as fout:

            for inst in instructions:
                with Pool(4) as p: 
                    rouge_scores = p.map(partial(scorer.score, inst), prompt_instructions)
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                
                if max(rouge_scores) > 0.7:
                    continue
                                
                most_similar_instructions = {
                        prompt_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                machine_gen_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    # "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")

            request_idx += 1

# if __name__ == "__main__":
    # instr = GenerateInstructions(seed_tasks_path="</path>", machine_gen_instructions_path="machine_generated_instructions.jsonl", instruction_dir=".")
    # for _ in range(4):
    #     instr.generate()