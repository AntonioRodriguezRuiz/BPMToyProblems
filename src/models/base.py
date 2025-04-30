import gc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from utils.logging_utils import setup_logger, log_exception


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
        # Initialize logger for the class
        self.logger = setup_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__} instance")

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
            self.logger.info(f"Manually loading model: {self.model_name}")
            try:
                self._loaded_model, self._loaded_processor = self.load_model()
                self._loaded = True
                self.logger.info(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                log_exception(self.logger, e, {"model_name": self.model_name})
                self.logger.error(f"Failed to load model: {self.model_name}")
                raise

    def manual_unload(self) -> None:
        if self._loaded:
            self.logger.info(f"Manually unloading model: {self.model_name}")
            try:
                self.unload(self._loaded_model, self._loaded_processor)
                self._loaded = False
                self.logger.info(f"Model unloaded successfully: {self.model_name}")
            except Exception as e:
                log_exception(self.logger, e, {"model_name": self.model_name})
                self.logger.warning(f"Error during model unloading: {self.model_name}")

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
        try:
            for arg in args:
                del arg
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug("Memory cleaned up and CUDA cache emptied")
        except Exception as e:
            log_exception(self.logger, e)
            self.logger.warning("Failed to clean up memory properly")

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
        self.logger.debug(f"Model call with prompt length: {len(prompt)}")
        try:
            result = self._call(
                prompt=prompt,
                stop=stop,
                callbacks=callbacks,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
            self.logger.debug(f"Model response generated (length: {len(result)})")
            return result
        except Exception as e:
            log_exception(self.logger, e, {
                "prompt_length": len(prompt),
                "model_name": self.model_name,
                "kwargs": str(kwargs.keys()),
            })
            self.logger.error(f"Model call failed: {self.model_name}")
            raise