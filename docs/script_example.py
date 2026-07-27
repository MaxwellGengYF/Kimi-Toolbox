from kimix import *

# Initialize the LLM client session.
# - model: the model to use (deepseek-v4-flash)
# - max_context_size: maximum context window (1M tokens)
# - capabilities: enables the "thinking" capability for chain-of-thought reasoning
# - url: API endpoint (DeepSeek API in this example)
# - type: API protocol type ("openai_legacy" for OpenAI-compatible endpoints)
# - api_key: authentication key (replace with your actual key)
# - thinking_effort: controls how much effort the model spends on reasoning ("max" = most thorough)
init(config_json='''
{
    "model": "deepseek-v4-flash",
    "max_context_size": 1048576,
    "capabilities": ["thinking"],
    "url": "https://api.deepseek.com",
    "type": "openai_legacy",
    "api_key": "sk-xxx",
    "thinking_effort": "max"
}
''')

# Send a prompt to the initialized LLM session.
# The agent processes the input and returns a response through the configured output handler.
prompt('hello!')
