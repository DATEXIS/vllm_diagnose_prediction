import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional

import httpx
from aiohttp import ClientSession, ClientTimeout
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

async def check_connection(api_base: str):
    """Checks if the vLLM server is accessible."""
    health_url = api_base.replace("/v1", "/health")
    logger.info(f"Checking connection to {health_url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=10.0)
            response.raise_for_status()
            logger.info("Successfully connected to vLLM server.")
    except Exception as e:
        logger.error(f"Could not connect to vLLM server at {health_url}: {e}")
        raise RuntimeError("vLLM server connection failed.")

def build_payload(config: dict, prompt: str, schema: Optional[Dict[str, Any]] = None) -> dict:
    """Builds the JSON payload for the vLLM API request."""
    model_name = config['model']['name']
    inf_cfg = config['inference']
    
    payload = {
        "model": model_name,
        "temperature": inf_cfg.get('temperature', 0.2),
        "max_tokens": inf_cfg.get('max_tokens', 2000),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    if inf_cfg.get('guided_decoding', False) and schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": schema
        }
    return payload

def build_coroutine(session: ClientSession, config: dict, url: str, prompt: str, schema: Optional[Dict[str, Any]] = None):
    payload = build_payload(config, prompt, schema)
    headers = {"Content-Type": "application/json"}

    async def request_coro():
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Request failed: {e}\n{traceback.format_exc()}")
            return None

    return request_coro

async def gather_with_concurrency(n: int, coros_with_prompts: List[tuple]):
    """Gathers coroutines with a concurrency limit using a Semaphore."""
    semaphore = asyncio.Semaphore(n)

    async def sem_wrapper(coro, prompt):
        async with semaphore:
            try:
                # Add a timeout to prevent hanging forever
                return await asyncio.wait_for(coro(), timeout=600)
            except Exception as e:
                logger.error(f"Task failed for prompt: {prompt[:50]}... Error: {e}")
                return None

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

def extract_text_from_responses(responses: List[Any]) -> List[str]:
    """Extracts the final answer string from the chat completions payload."""
    final_output = []
    for i, resp in enumerate(responses):
        if not resp or "choices" not in resp:
            logger.warning(f"Response {i}: Missing 'choices' field - {resp}")
            final_output.append("")
            continue
        try:
            text = resp["choices"][0]["message"]["content"]
            logger.debug(f"Response {i} (first 500ch): {text[:500]}")
            logger.debug(f"Response {i} (last 500ch): {text[-500:]}")
            final_output.append(text)
        except (KeyError, IndexError) as e:
            logger.warning(f"Response {i}: Failed to extract text - {e}, resp: {resp}")
            final_output.append("")
    return final_output
