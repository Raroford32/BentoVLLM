import uuid
from typing import AsyncGenerator

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated


MAX_TOKENS = 1024
PROMPT_TEMPLATE = """<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
"""


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model='google/gemma-7b-it', max_model_len=MAX_TOKENS))

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SamplingParams(max_tokens=max_tokens))

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
