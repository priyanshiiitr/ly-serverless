import os
import boto3
import logging
from openai import OpenAI, BadRequestError

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # For CloudWatch visibility

# Ensure handlers are added only once in Lambda environment
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# AWS + OpenAI clients
s3 = boto3.client("s3")
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
BASE_URL = "https://api.groq.com/openai/v1" if PROVIDER == "groq" else "https://api.openai.com/v1"
API_KEY = os.getenv("GROQ_API_KEY") if PROVIDER == "groq" else os.getenv("OPENAI_API_KEY")

if PROVIDER == "groq":
    default_model = "llama-3.3-70b-versatile"
    fallback_models = ["llama-3.1-8b-instant"]
else:
    default_model = "gpt-4o-mini"
    fallback_models = ["gpt-3.5-turbo"]

MODEL = os.getenv("LLM_MODEL", default_model)
MODEL_CANDIDATES = [MODEL] + [m for m in fallback_models if m != MODEL]
BUCKET = os.getenv("BUCKET_NAME", "lambda-genai-bucket-3")

logger.info(f"LLM Provider: {PROVIDER}, Model: {MODEL}, Bucket: {BUCKET}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def _is_decommissioned_model_error(error):
    message = str(error).lower()
    return "model_decommissioned" in message or "decommissioned" in message or "no longer supported" in message

def summarize_text(text):
    logger.info("Starting text summarization. Text length: %d characters", len(text))
    last_error = None

    for index, model_name in enumerate(MODEL_CANDIDATES):
        try:
            if index > 0:
                logger.warning("Retrying with fallback model: %s", model_name)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful text summarizer."},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ],
                temperature=0.5,
                max_tokens=300
            )
            logger.info("Summarization completed with model: %s", model_name)
            return response.choices[0].message.content
        except BadRequestError as e:
            last_error = e
            if _is_decommissioned_model_error(e) and index < len(MODEL_CANDIDATES) - 1:
                logger.warning("Model %s is unavailable. Trying next model.", model_name)
                continue

            logger.error("Failed during summarization: %s", str(e), exc_info=True)
            raise
        except Exception as e:
            logger.error("Failed during summarization: %s", str(e), exc_info=True)
            raise

    logger.error("All configured models failed: %s", MODEL_CANDIDATES)
    if last_error:
        raise last_error
    raise RuntimeError("Summarization failed and no model could be used.")

def lambda_handler(event, context):
    logger.info("Lambda triggered. Raw event: %s", event)

    try:
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        logger.info("Received file: s3://%s/%s", bucket_name, key)

        if not key.endswith(".txt"):
            logger.warning("Skipping non-text file: %s", key)
            return {"status": "skipped_non_txt"}

        file_obj = s3.get_object(Bucket=bucket_name, Key=key)
        text = file_obj["Body"].read().decode("utf-8")
        logger.info("Downloaded file. Size: %d characters", len(text))

        summary = summarize_text(text)

        filename = key.split("/")[-1]
        output_key = f"output/summary_{filename}"
        s3.put_object(Body=summary.encode("utf-8"), Bucket=bucket_name, Key=output_key)

        logger.info("Successfully saved summary to: s3://%s/%s", bucket_name, output_key)
        return {"status": "complete"}

    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        raise
