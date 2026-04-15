# LY-Serverless-Deployment
This is a mini project to demonstrate serverless deployment on AWS.

## LLM model configuration

- `LLM_PROVIDER` supports `groq` (default) and `openai`.
- `LLM_MODEL` is optional. If omitted, the app uses a provider-specific default.
- If a selected model is deprecated (for example, Groq model decommission), the code retries with a fallback model automatically.
