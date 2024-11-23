import base64
from pathlib import Path
from typing import Union
from openai import OpenAI

AVOID_API_CALLS = True

def ask_question_about_image(image_path: Union[str, Path], question: str) -> str:
    if AVOID_API_CALLS:
        return open("data/example_response.txt", "r").read()
    
    def encode_image(image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    assert isinstance(base64_image, str)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
    )

    with open("response.txt", "w") as file:
        file.write(str(response))

    return response.choices[0].message.content
