import base64
import json
from pathlib import Path
from typing import Union
from openai import OpenAI
from datetime import datetime

from tqdm import tqdm

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


def transcribe_image(image_path: Union[str, Path]) -> str:
    """
    Transcribe the text in the image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        The transcribed text.
    """
    Path("data/responses/transcriptions/").mkdir(parents=True, exist_ok=True)
    question = "Please transcribe the text in the image."
    answer = ask_question_about_image(image_path, question)

    with open(f"data/responses/transcriptions/{image_path.name}.txt", "w") as file:
        file.write(answer)
    return answer


if __name__ == "__main__":
    for image_path in tqdm(Path("data/recipe_images").glob("*.jpg"), desc="Transcribing images in data/recipe_images"):
        response_message = transcribe_image(image_path)

        print(response_message)
