import json
import torch
from typing import List, Optional, Any, Dict, Sequence, Union, Callable

from pydantic import PrivateAttr, ConfigDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig, BitsAndBytesConfig

from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


def _lc_messages_to_qwen(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = getattr(m, "type", None)
        content = str(getattr(m, "content", ""))
        if role == "human":
            out.append({"role": "user", "content": content})
        elif role == "ai":
            out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content})
        else:
            out.append({"role": "user", "content": content})
    return out


def _safe_tool_schema_text(tool_obj: Any) -> str:
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return ""
    if hasattr(args_schema, "model_json_schema"):
        try:
            return json.dumps(args_schema.model_json_schema(), ensure_ascii=False)
        except Exception:
            pass
    if hasattr(args_schema, "schema"):
        try:
            return json.dumps(args_schema.schema(), ensure_ascii=False)
        except Exception:
            pass
    return ""


class QwenHFChat(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 800
    temperature: float = 0.0
    device_map: str = "auto"

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _llm: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )

        self._model.generation_config.max_length = None
        self._model.generation_config.max_new_tokens = self.max_new_tokens
        self._model.generation_config.pad_token_id = self._tokenizer.eos_token_id
        self._model.generation_config.do_sample = (self.temperature > 0)

        if self.temperature > 0:
            self._model.generation_config.temperature = self.temperature

        gen_pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            return_full_text=False,
            # Thêm 2 dòng này:
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        self._llm = HuggingFacePipeline(pipeline=gen_pipe)

    @property
    def _llm_type(self) -> str:
        return "qwen_hf_chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "device_map": self.device_map,
        }

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
            tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
            **kwargs: Any
    ) -> Runnable:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any
    ) -> ChatResult:
        qwen_msgs = _lc_messages_to_qwen(messages)
        tools = kwargs.get("tools", None)
        tool_choice = kwargs.get("tool_choice", None)

        if tools:
            tool_descs = []
            for t in tools:
                name = getattr(t, "name", None) or getattr(t, "__name__", "unknown_tool")
                desc = getattr(t, "description", "") or ""
                schema_text = _safe_tool_schema_text(t)
                block = f"- Tool name: {name}\n  Description: {desc}"
                if schema_text:
                    block += f"\n  Args schema: {schema_text}"
                tool_descs.append(block)

            tool_policy = (
                    "You may use tools if needed.\n"
                    "Available tools:\n" + "\n".join(tool_descs) + "\n\n"
                                                                   "If a tool is needed, respond ONLY with valid JSON in this exact format:\n"
                                                                   '{"tool_name": "<name>", "arguments": {...}}\n'
                                                                   "Do not include markdown fences.\n"
                                                                   "If no tool is needed, answer normally."
            )
            if tool_choice:
                tool_policy += f"\nTool choice preference: {tool_choice}"

            qwen_msgs = [{"role": "system", "content": tool_policy}] + qwen_msgs

        prompt = self._tokenizer.apply_chat_template(
            qwen_msgs,
            tokenize=False,
            add_generation_prompt=True
        )

        text = self._llm.invoke(prompt)

        if stop and isinstance(text, str):
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]

        gen = ChatGeneration(message=AIMessage(content=str(text)))
        return ChatResult(generations=[gen])


if __name__ == "__main__":
    llm = QwenHFChat(
        # model_name="Qwen/Qwen3-8B",
        # model_name="Qwen/Qwen2.5-7B-Instruct",
        # model_name="Qwen/Qwen2.5-3B-Instruct",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.0,
        max_new_tokens=300,
    )

    response = llm.invoke([
        SystemMessage(content="Bạn là trợ lý AI trả lời ngắn gọn, rõ ràng bằng tiếng Việt."),
        HumanMessage(content="Giải thích overfitting trong machine learning cho người mới học.")
    ])

    print(response.content)