import base64
import json
from pathlib import Path
from typing import Union
from openai import OpenAI
from datetime import datetime

AVOID_API_CALLS = False

def ask_question_about_image(image_path: Union[str, Path], question: str) -> str:
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    
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

    response_info = {
        "question": question,
        "image_name": image_path.name,
        "response": response.to_dict(),
        "answer": response.choices[0].message.content
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
    with open(f"data/responses/{timestamp}_response.json", "w") as file:
        json.dump(response_info, file)
    return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "data/recipe_images/PXL_20241123_132750395.jpg"
    question = "Please transcribe the text in the image."
    response = ask_question_about_image(image_path, question)
    print(response)
