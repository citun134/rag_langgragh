import json
import requests

from typing import List, Optional, Any, Dict, Sequence, Union, Callable

from pydantic import PrivateAttr, ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
import json
import requests

from typing import List, Optional, Any, Dict, Sequence, Union, Callable

from pydantic import PrivateAttr, ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class _RemoteGenerationConfig:
    """
    Compatibility object.

    This makes remote API models behave like local HuggingFace models
    for old code that accesses:
        llm._model.generation_config.max_new_tokens
    """

    def __init__(self, max_new_tokens: int = 300, temperature: float = 0.0):
        self.max_length = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = temperature > 0
        self.pad_token_id = None
        self.eos_token_id = None


class _RemoteModelProxy:
    """
    Fake model object for compatibility with old local-HF code.
    """

    def __init__(self, max_new_tokens: int = 300, temperature: float = 0.0):
        self.generation_config = _RemoteGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def _lc_messages_to_api(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Convert LangChain messages to API messages.
    """

    api_messages = []

    for message in messages:
        role = getattr(message, "type", None)
        content = str(getattr(message, "content", ""))

        if role == "human":
            api_messages.append({
                "role": "user",
                "content": content,
            })
        elif role == "ai":
            api_messages.append({
                "role": "assistant",
                "content": content,
            })
        elif role == "system":
            api_messages.append({
                "role": "system",
                "content": content,
            })
        else:
            api_messages.append({
                "role": "user",
                "content": content,
            })

    return api_messages


def _safe_tool_schema_text(tool_obj: Any) -> str:
    """
    Safely convert LangChain tool schema to JSON text.
    """

    args_schema = getattr(tool_obj, "args_schema", None)

    if args_schema is None:
        return ""

    if hasattr(args_schema, "model_json_schema"):
        try:
            return json.dumps(
                args_schema.model_json_schema(),
                ensure_ascii=False,
            )
        except Exception:
            pass

    if hasattr(args_schema, "schema"):
        try:
            return json.dumps(
                args_schema.schema(),
                ensure_ascii=False,
            )
        except Exception:
            pass

    return ""


class KaggleAPIChat(BaseChatModel):
    """
    LangChain-compatible chat model.

    This class does not load the model locally.
    It sends requests to the Kaggle FastAPI server.

    It also exposes:
        self._model.generation_config.max_new_tokens

    so old code written for HuggingFace local models will not crash.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    api_url: str
    api_key: str
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 300
    temperature: float = 0.0
    timeout: int = 180

    _headers: Dict[str, str] = PrivateAttr(default_factory=dict)

    # Compatibility private attrs, matching the old QwenHFChat style
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _llm: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)

        self.api_url = self.api_url.rstrip("/")

        self._headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        # Fake local model object, so old code can do:
        # llm._model.generation_config.max_new_tokens = max_tokens
        self._model = _RemoteModelProxy(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

    @property
    def _llm_type(self) -> str:
        return "kaggle_api_chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable:
        return self.bind(
            tools=list(tools),
            tool_choice=tool_choice,
            **kwargs,
        )

    def _get_effective_max_new_tokens(self, kwargs: Dict[str, Any]) -> int:
        """
        Priority:
        1. kwargs["max_new_tokens"]
        2. self._model.generation_config.max_new_tokens
        3. self.max_new_tokens
        """

        if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
            return int(kwargs["max_new_tokens"])

        generation_config = getattr(self._model, "generation_config", None)

        if generation_config is not None:
            value = getattr(
                generation_config,
                "max_new_tokens",
                self.max_new_tokens,
            )
            return int(value)

        return int(self.max_new_tokens)

    def _get_effective_temperature(self, kwargs: Dict[str, Any]) -> float:
        """
        Priority:
        1. kwargs["temperature"]
        2. self._model.generation_config.temperature
        3. self.temperature
        """

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            return float(kwargs["temperature"])

        generation_config = getattr(self._model, "generation_config", None)

        if generation_config is not None:
            value = getattr(
                generation_config,
                "temperature",
                self.temperature,
            )
            return float(value)

        return float(self.temperature)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = _lc_messages_to_api(messages)

        tools = kwargs.get("tools", None)
        tool_choice = kwargs.get("tool_choice", None)

        if tools:
            tool_descs = []

            for tool in tools:
                name = (
                    getattr(tool, "name", None)
                    or getattr(tool, "__name__", "unknown_tool")
                )
                desc = getattr(tool, "description", "") or ""
                schema_text = _safe_tool_schema_text(tool)

                block = f"- Tool name: {name}\n  Description: {desc}"

                if schema_text:
                    block += f"\n  Args schema: {schema_text}"

                tool_descs.append(block)

            tool_policy = (
                "You may use tools if needed.\n"
                "Available tools:\n"
                + "\n".join(tool_descs)
                + "\n\n"
                "If a tool is needed, respond ONLY with valid JSON in this exact format:\n"
                '{"tool_name": "<name>", "arguments": {...}}\n'
                "Do not include markdown fences.\n"
                "If no tool is needed, answer normally."
            )

            if tool_choice:
                tool_policy += f"\nTool choice preference: {tool_choice}"

            api_messages = [
                {
                    "role": "system",
                    "content": tool_policy,
                }
            ] + api_messages

        max_new_tokens = self._get_effective_max_new_tokens(kwargs)
        temperature = self._get_effective_temperature(kwargs)

        # Keep public fields synced with compatibility object
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if self._model is not None:
            self._model.generation_config.max_new_tokens = max_new_tokens
            self._model.generation_config.temperature = temperature
            self._model.generation_config.do_sample = temperature > 0

        payload = {
            "messages": api_messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop": stop,
        }

        try:
            response = requests.post(
                self.api_url + "/v1/chat",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            error_text = ""

            try:
                error_text = response.text
            except Exception:
                pass

            raise RuntimeError(
                f"Kaggle API request failed: {exc}\n"
                f"URL: {self.api_url}/v1/chat\n"
                f"Response: {error_text}"
            ) from exc

        data = response.json()
        text = str(data.get("content", ""))

        generation = ChatGeneration(
            message=AIMessage(content=text)
        )

        return ChatResult(
            generations=[generation]
        )


if __name__ == "__main__":
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = KaggleAPIChat(
        api_url="https://YOUR_PINGGY_URL_HERE",
        api_key="my-secret-api-key-123",
        temperature=0.0,
        max_new_tokens=100,
    )

    # Compatibility test
    llm._model.generation_config.max_new_tokens = 50
    print("Compatibility max_new_tokens:", llm._model.generation_config.max_new_tokens)

    response = llm.invoke([
        SystemMessage(content="Bạn là trợ lý AI trả lời ngắn gọn bằng tiếng Việt."),
        HumanMessage(content="Giải thích overfitting trong machine learning.")
    ])

    print(response.content)
if __name__ == "__main__":

    from langchain_core.messages import SystemMessage, HumanMessage
    link_api = "https://pxpmp-104-154-27-43.run.pinggy-free.link/"
    llm = KaggleAPIChat(
        api_url=link_api,
        api_key="my-secret-api-key-123",
        temperature=0.0,
        max_new_tokens=300,
    )

    response = llm.invoke([
        SystemMessage(
            content="Bạn là trợ lý AI trả lời ngắn gọn, rõ ràng bằng tiếng Việt."
        ),
        HumanMessage(
            content="Giải thích overfitting trong machine learning cho người mới học."
        )
    ])

    print(response.content)