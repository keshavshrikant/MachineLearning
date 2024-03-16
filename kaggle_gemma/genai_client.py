import os

from openai import OpenAI
from openai import AuthenticationError
from constants import WORDS_TO_IGNORE_IN_GENERATION

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), 
       stop=stop_after_attempt(6)
    )
def generate_text(openai_client, prompt, 
                max_tokens=2000,
                temperature=0.7,
                top_p=0.92,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],                    
                n=1):


    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},        
        ],
        temperature=temperature,
        stop=stop_sequences,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,        
    )

    try:
        model_output = response.choices[0].message.content
        if not isinstance(model_output, str):
            raise TypeError(f"Bad type for generation output: {type(model_output)}. Can only be string.")
    except (KeyError, TypeError) as e:
        print("Bad data from endpoint")
        raise e
    except AuthenticationError as e:
        print("Token Expired, figre out how to create a new one!")
    return model_output




