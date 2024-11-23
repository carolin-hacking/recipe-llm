from openai import OpenAI

from src.transcription import ask_question_about_image


def test_openai_key_works():
    """
    From https://platform.openai.com/docs/quickstart?language-preference=python
    Please set the env var OPENAI_API_KEY to your API key.
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming.",
            },
        ],
    )

    message: str = completion.choices[0].message.content
    assert isinstance(message, str)
    
    
def test_openai_request_with_image():
    question = "What is in this image?"
    image_path = "data/recipe_images/PXL_20241123_132750395.jpg"
    
    response = ask_question_about_image(image_path, question)
    assert isinstance(response, str)
    assert response != ""
