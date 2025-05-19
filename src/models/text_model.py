from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

from models.base import ModelInterface


class TextModel(ModelInterface):
    """
    A model interface for text-only inference.

    **Example Usage (Local Inference):**

    ```python
    from models.models import TextModel

    # Using a locally loaded model
    model = TextModel("path/to/local/text-model")
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import TextModel

    # Configure for OpenAI API inference
    model = TextModel("your-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```
    """

    capabilities: List[str] = ["text"]

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Performs an inference using a text-only model.
        """
        assert isinstance(sys_prompt, str), "sys_prompt must be a string"
        assert isinstance(user_prompt, str), "user_prompt must be a string"
        assert isinstance(max_tokens, int), "max_tokens must be an integer"

        messages: List[Dict[str, Any]] = []
        processed_output_text: str
        if self.openai_server:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            client = OpenAI(base_url=self.openai_server, api_key=self.api_key)
            processed_output_text = (
                client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                .choices[0]
                .message.content
            )
        else:
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            processed_output_text = self._local_inference(
                messages, max_tokens=max_tokens, **kwargs
            )
        if "result_queue" in kwargs:
            kwargs["result_queue"].put(processed_output_text)
            return None
        else:
            return processed_output_text

    def _local_inference(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        if self.loaded:
            model: AutoModelForCausalLM
            processor: AutoProcessor
            model, processor = self._loaded_model, self._loaded_processor
        else:
            model, processor = self.load_model()

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            padding=True if "padding" not in kwargs else kwargs["padding"],
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generate output tokens
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        processed_output_text: str = next(
            iter(
                processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=self.skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            ),
            "",
        )

        if self.loaded:
            del model
            del processor
        del inputs
        del generated_ids
        del generated_ids_trimmed

        torch.cuda.empty_cache()

        return processed_output_text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Wraps inference in a separate process to help free GPU memory.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert stop is None or isinstance(stop, list), "stop must be None or a list"
        max_tokens: int = kwargs.pop("max_tokens", 2048)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        if self.loaded:
            result = self.inference(sys_prompt, prompt, max_tokens, **kwargs)
        else:
            result_queue: Queue = Queue()

            kwargs["result_queue"] = result_queue
            p = Process(
                target=self.inference,
                args=(sys_prompt, prompt, max_tokens),
                kwargs=kwargs,
            )
            p.start()
            p.join()

            result = result_queue.get()
        assert isinstance(result, str), "result must be a string"
        return result

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        """
        Loads and returns the text-only model and processor.
        """
        assert isinstance(self.model_name, str), "model_name must be a string"
        # Check if we have more than one GPU
        # if torch.cuda.device_count() > 1:
        #     rank = int(os.environ["RANK"])
        #     device = torch.device(f"cuda:{rank}")
        #     torch.cuda.set_device(device)
        #     torch.distributed.init_process_group("nccl", device_id=device)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor
