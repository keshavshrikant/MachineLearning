GENERATE_INTRUCTIONS_SYS_MSG = """Following are 8 tasks that you have. Your job is to generate 8 more."""
GENERATE_TRANSFORMATION_SYS_MSG = """Provided below is a creative writing paragraph and an insturction for rewriting it in a certain manner. Please follow the insturction to the tee and reqrite the paragraph as specified. 

Paragraph:
{para}

Instruction:
{instruction}

Always respond with a JSON like: 
{{
    "Rewritten Paragraph": rewritten paragraph
}}
"""

WORDS_TO_IGNORE_IN_GENERATION = ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]


FINETUNE_TEMPLATES = {
    "gemma": {
        "description": "Template created for Kaggle Competition V1.",
        "prompt_input": """Provided below are two essays. The rewritten essay was created from the Original essay via a prompt. Your job is to figure out how the Original essay was transformed into the Rewritten essay. Analyze and identify the changes in style, tone, structure, point of view, theme, genre of the Rewritten essay vis-a-vis Orignial essay and figure out the prompt which would have guided the transformation. Only give me the PROMPT. Start directly with the prompt, that's all I need. Output should be one line ONLY.
        
    Original Essay:
    {original_text}

    Rewritten Essay:
    {rewritten_text}
        
    Please come up with a prompt that must have been used to guide the transformation from the original to the rewritten essay. Only give me the PROMPT. Start directly with the prompt, that's all I need. Output should be only line ONLY.

    ### Prompt: 
    """,    
        "response_split": "### Prompt:"    
    }
}