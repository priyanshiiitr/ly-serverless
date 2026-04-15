import os
from openai import OpenAI, BadRequestError

# Choose provider: "groq" or "openai"
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # default: groq

if PROVIDER == "groq":
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY = os.getenv("GROQ_API_KEY")
    default_model = "llama-3.3-70b-versatile"
    fallback_models = ["llama-3.1-8b-instant"]
else:
    BASE_URL = "https://api.openai.com/v1"
    API_KEY = os.getenv("OPENAI_API_KEY")
    default_model = "gpt-4o-mini"
    fallback_models = ["gpt-3.5-turbo"]

MODEL = os.getenv("LLM_MODEL", default_model)
MODEL_CANDIDATES = [MODEL] + [m for m in fallback_models if m != MODEL]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

INPUT_FOLDER = "input_files"
OUTPUT_FOLDER = "output_summaries"

def _is_decommissioned_model_error(error):
    message = str(error).lower()
    return "model_decommissioned" in message or "decommissioned" in message or "no longer supported" in message

def summarize_text(text):
    last_error = None

    for index, model_name in enumerate(MODEL_CANDIDATES):
        try:
            if index > 0:
                print(f"Primary model unavailable, retrying with fallback: {model_name}")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful text summarizer."},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            last_error = e
            if _is_decommissioned_model_error(e) and index < len(MODEL_CANDIDATES) - 1:
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Summarization failed and no model could be used.")

def summarize_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return summarize_text(text)

def process_files():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"summary_{filename}")

            summary = summarize_text_from_file(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

            print(f"Summarized: {filename} -> {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    process_files()
