import gc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class ModelInterface(LLM, ABC):
    """
    Abstract base class for model interfaces.
    """

    model_name: Optional[str] = None
    capabilities: List[str] = []
    skip_special_tokens: bool = False
    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 0
    openai_server: Optional[str] = None
    api_key: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loaded: bool = False
        self._loaded_model: Optional[Any] = None
        self._loaded_processor: Optional[Any] = None

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__

    @property
    def _history_len(self) -> int:
        return self.history_len

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len,
        }

    @property
    def loaded(self) -> bool:
        return self._loaded

    def manual_load(self) -> None:
        if not self._loaded:
            self._loaded_model, self._loaded_processor = self.load_model()
            self._loaded = True

    def manual_unload(self) -> None:
        if self._loaded:
            self.unload(self._loaded_model, self._loaded_processor)
            self._loaded = False

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input.

        Performs an inference using qwen-vl based models.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            str: The model output as a string. SHOULD NOT include the prompt.
        """
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, ...]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Any, ...]: The loaded model and any additional components.
        """
        pass

    def unload(self, *args: Any) -> None:
        """
        Unloads the given items and extras from cuda memory.

        Args:
            *args (Any): The items to unload.
        """
        for arg in args:
            del arg
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(
        self, prompt, stop=None, callbacks=None, *, tags=None, metadata=None, **kwargs
    ) -> str:
        """
        Calls the model with the given prompt and additional arguments.

        Args:
            prompt (str): The prompt to generate from.
            stop (Optional[List[str]]): Stop words to use when generating.
            callbacks (Optional[Any]): Callbacks for the run.
            tags (Optional[Any]): Tags for the run.
            metadata (Optional[Any]): Metadata for the run.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: The model output as a string.
        """
        result = self._call(
            prompt=prompt,
            stop=stop,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        return result