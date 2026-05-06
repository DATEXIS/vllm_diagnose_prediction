import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional

import httpx
import traceback
from aiohttp import ClientSession, ClientTimeout
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_vllm_connected = False

async def check_connection(api_base: str):
    """Checks if the vLLM server is accessible."""
    global _vllm_connected
    if _vllm_connected:
        return
        
    health_url = api_base.replace("/v1", "/health")
    backoff_time = 1
    num_tries = 0
    max_tries = 10000
    while num_tries <= max_tries:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=10.0)
                response.raise_for_status()
                logger.debug("Successfully connected to vLLM server.")
                _vllm_connected = True
                return
        except Exception as e:
            logger.debug(f"Connect {num_tries} to {health_url}, "
                        f"retrying in {backoff_time}s: {e}")
            await asyncio.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, 60)
            num_tries += 1
    raise RuntimeError(f"Could not connect to vLLM server after {max_tries} attempts")

def build_payload(config: dict, prompt: str, schema: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None) -> dict:
    """Builds the JSON payload for the vLLM API request."""
    model_name = config['model']['name']
    inf_cfg = config['inference']
    
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    payload = {
        "model": model_name,
        "temperature": inf_cfg.get('temperature', 0.2),
        "max_tokens": inf_cfg.get('max_tokens', 2000),
        "messages": messages,
        "stream": False,
    }

    if "reasoning_budget" in inf_cfg:
        payload["reasoning_budget"] = inf_cfg["reasoning_budget"]

    if inf_cfg.get('guided_decoding', False) and schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema
        }
    return payload

def build_coroutine(session: ClientSession, config: dict, url: str, prompt: str, schema: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None):
    payload = build_payload(config, prompt, schema, system_prompt)
    headers = {"Content-Type": "application/json"}

    async def request_coro():
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Request failed with status {response.status}: {text}")
            return await response.json()

    return request_coro

async def gather_with_concurrency(n: int, coros_with_prompts: List[tuple]):
    """Gathers coroutines with a concurrency limit using a Semaphore."""
    semaphore = asyncio.Semaphore(n)

    async def sem_wrapper(coro, prompt):
        async with semaphore:
            return await asyncio.wait_for(coro(), timeout=600)

    wrapped = [sem_wrapper(coro, prompt) for coro, prompt in coros_with_prompts]
    return await tqdm_asyncio.gather(*wrapped, desc="Processing Prompts")

async def run_inference(config: dict, prompts: List[str], schema: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Runs concurrent inference over all prompts."""
    job_name = config.get('job_name', 'default')
    namespace = config.get('k8s', {}).get('namespace', 'default')
    
    # Check if a manual override is given (for local testing), otherwise build dynamically
    api_base = config['model'].get('api_base')
    if not api_base:
        api_base = f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"
        
    await check_connection(api_base)
    
    url = f"{api_base}/chat/completions"
    concurrency = config['inference'].get('concurrency', 10)
    
    # Hide the "for 1 prompts" message to declutter logs
    if len(prompts) > 1:
        logger.info(f"Starting inference with concurrency={concurrency} for {len(prompts)} prompts...")
    
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        coroutines = [
            build_coroutine(session, config, url, prompt, schema)
            for prompt in prompts
        ]
        # Zip coroutines and prompts so we can track context during execution
        coros_with_prompts = list(zip(coroutines, prompts))
        responses = await gather_with_concurrency(concurrency, coros_with_prompts)
        
    return extract_text_from_responses(responses)

def build_payload_from_messages(
    config: dict,
    messages: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a vLLM payload from a pre-constructed messages list.

    Mirrors build_payload but accepts an explicit messages list instead of a
    flat prompt string + optional system_prompt. Used by the Generator so it
    can send system / user / assistant roles separately.
    """
    model_name = config['model']['name']
    inf_cfg = config['inference']

    payload = {
        "model": model_name,
        "temperature": inf_cfg.get('temperature', 0.2),
        "max_tokens": inf_cfg.get('max_tokens', 2000),
        "messages": messages,
        "stream": False,
    }

    if "reasoning_budget" in inf_cfg:
        payload["reasoning_budget"] = inf_cfg["reasoning_budget"]

    if "repetition_penalty" in inf_cfg:
        payload["repetition_penalty"] = inf_cfg["repetition_penalty"]

    if inf_cfg.get('guided_decoding', False) and schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }
    return payload


def build_coroutine_from_messages(
    session: ClientSession,
    config: dict,
    url: str,
    messages: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]] = None,
):
    """Return an async coroutine factory for a single messages-list request."""
    payload = build_payload_from_messages(config, messages, schema)
    headers = {"Content-Type": "application/json"}

    async def request_coro():
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Request failed with status {response.status}: {text}")
            return await response.json()

    return request_coro


async def run_inference_messages(
    config: dict,
    messages_list: List[List[Dict[str, Any]]],
    schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run concurrent inference from pre-built message lists.

    Drop-in replacement for run_inference when the caller already has
    role-separated messages (system / user / assistant) rather than a flat
    prompt string. Used by the Generator to send proper chat roles and to
    pre-fill the assistant turn with the <coding_review> block.
    """
    job_name = config.get('job_name', 'default')
    namespace = config.get('k8s', {}).get('namespace', 'default')

    api_base = config['model'].get('api_base')
    if not api_base:
        api_base = f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"

    await check_connection(api_base)

    url = f"{api_base}/chat/completions"
    concurrency = config['inference'].get('concurrency', 10)

    if len(messages_list) > 1:
        logger.info(
            f"Starting inference with concurrency={concurrency} for {len(messages_list)} prompts..."
        )

    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        coroutines = [
            build_coroutine_from_messages(session, config, url, messages, schema)
            for messages in messages_list
        ]
        coros_with_prompts = list(zip(coroutines, messages_list))
        responses = await gather_with_concurrency(concurrency, coros_with_prompts)

    return extract_text_from_responses(responses)


async def run_inference_with_system(
    config: dict,
    prompts: List[str],
    system_prompt: str = "You are a helpful medical assistant.",
    temperature: float = 0.4,
    max_tokens: int = 2000,
) -> List[Optional[str]]:
    """Runs concurrent inference with a system prompt, returns raw text responses."""
    job_name = config.get('job_name', 'default')
    namespace = config.get('k8s', {}).get('namespace', "default")
    
    api_base = config['model'].get('api_base')
    if not api_base:
        api_base = f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"
        
    url = f"{api_base}/chat/completions"
    concurrency = config.get('inference', {}).get('concurrency', 10)
    
    async with ClientSession(timeout=ClientTimeout(total=3600)) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def send_with_sem(prompt: str):
            async with semaphore:
                payload = {
                    "model": config['model'].get('name', 'Qwen/Qwen3-8B'),
                    "temperature": temperature,
                    "n": 1,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "echo": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                }
                headers = {"Content-Type": "application/json"}
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                            logger.error(f"Error {resp.status}: {await resp.text()}")
                            return None
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    return None

        tasks = [send_with_sem(p) for p in prompts]
        from tqdm.asyncio import tqdm_asyncio
        return await tqdm_asyncio.gather(*tasks, desc="Processing")


def extract_text_from_responses(responses: List[Any]) -> List[str]:
    final_output = []
    for i, resp in enumerate(responses):
        if not resp:
            raise RuntimeError(f"Response {i} is None/empty")

        if "error" in resp:
            raise RuntimeError(f"Response {i} returned API error: {resp['error']}")

        if "choices" not in resp:
            if "reasoning_content" in resp:
                final_output.append(resp["reasoning_content"])
                continue

            raise RuntimeError(f"Response {i} missing 'choices' field. Keys: {list(resp.keys())}")

        try:
            # Standard path
            choice = resp["choices"][0]
            text = choice["message"].get("content", "")

            # 0.19.1 specific: check if content is null but reasoning exists
            if not text and "reasoning_content" in choice["message"]:
                text = choice["message"]["reasoning_content"]

            final_output.append(text)
        except (KeyError, IndexError) as e:
            logger.warning(f"Response {i}: Extraction failed - {e}")
            final_output.append("")
    return final_output
