import base64
import json
from pathlib import Path
from typing import List, Union
from openai import OpenAI
from datetime import datetime

from tqdm import tqdm

AVOID_API_CALLS = False

def get_image_content_item(image_path: Union[str, Path]) -> str:
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    
    
    def encode_image(image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    assert isinstance(base64_image, str)
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}


def ask_question_about_images(image_paths: Union[List[Union[str, Path]], Union[str, Path]], question: str) -> str:
    """
    Ask a question about multiple images.
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    *[get_image_content_item(image_path) for image_path in image_paths]
                ]
            }
        ]
    )

    response_info = {
        "question": question,
        "image_names": [image_path.name for image_path in image_paths],
        "response": response.to_dict(),
        "answer": response.choices[0].message.content
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
    with open(f"data/responses/{timestamp}_response.json", "w") as file:
        json.dump(response_info, file)
    return response.choices[0].message.content


def transcribe_image_set(image_paths: Union[List[Union[str, Path]], Union[str, Path]], output_file_name: str) -> str:
    """
    Transcribe the text in the image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        The transcribed text.
    """
    Path("data/responses/transcriptions/").mkdir(parents=True, exist_ok=True)
    question = "Please transcribe the text in the images."
    answer = ask_question_about_images(image_paths, question)

    with open(f"data/responses/transcriptions/{output_file_name}.txt", "w") as file:
        file.write(answer)
    return answer


if __name__ == "__main__":
    for recipe_directory in Path("data/recipe_images").glob("*"):
        if recipe_directory.is_dir():
            images_of_recipe = list(Path(recipe_directory).glob("*.jpg"))
            transcribe_image_set(images_of_recipe, f"recipe_{recipe_directory.name}")
