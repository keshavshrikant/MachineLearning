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