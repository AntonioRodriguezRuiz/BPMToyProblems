from torch.multiprocessing import Process, Queue, set_start_method
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from PIL import Image

from models.base import ModelInterface
import utils.images as images_utils


class VisionModel(ModelInterface):
    """
    A model interface for vision+text inference.

    **Example Usage (Local Inference):**

    ```python
    from models.models import VisionModel

    # Using a locally loaded model
    model = TextModel("hugginface/identifier")
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import VisionModel

    # Configure for OpenAI API inference
    model = TextModel("your-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    output = model("Hello world!", sys_prompt="Provide a short greeting")
    print(output)
    ```
    """

    capabilities: List[str] = ["image", "text"]

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
        Performs an inference using a vision+text model.
        """
        assert isinstance(sys_prompt, str), "sys_prompt must be a string"
        assert isinstance(user_prompt, str), "user_prompt must be a string"
        assert isinstance(max_tokens, int), "max_tokens must be an integer"

        messages: List[Dict[str, Any]] = []
        processed_output_text: str
        if self.openai_server:
            # if sys_prompt:
            #     # messages.append({"role": "system", "content": sys_prompt}) # Not all models support system messages
            #     messages.append({"role": "user", "content": sys_prompt})
            sys_prompt = sys_prompt if sys_prompt else "You are a helpful assistant"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{sys_prompt}<image>"},
                        *[
                            {
                                "type": t,
                                t: val,
                            }
                            if t != "image"
                            else {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_utils.im_2_b64(val)}"
                                },
                            }
                            for t, val in kwargs.items()
                            if t in self.capabilities
                        ],
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )
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
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": t,
                                t: val
                                if t != "image"
                                else images_utils.resize_img(val),
                            }
                            for t, val in kwargs.items()
                            if t in self.capabilities
                        ],
                        {"type": "text", "text": user_prompt},
                    ],
                }
            )
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
            model: AutoModel
            processor: AutoProcessor
            model, processor = self._loaded_model, self._loaded_processor
        else:
            model, processor = self.load_model()

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=151645,
            )

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
            self.unload(model, processor)
        self.unload(model, processor)
        return processed_output_text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Wraps vision+text inference in a separate process.
        """
        assert isinstance(prompt, str), "prompt must be a string"
        assert stop is None or isinstance(stop, list), "stop must be None or a list"
        max_tokens: int = kwargs.pop("max_tokens", 2048)
        sys_prompt: str = kwargs.pop("sys_prompt", "")

        if self.loaded:
            result = self.inference(sys_prompt, prompt, max_tokens, **kwargs)
        else:
            set_start_method("spawn")
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

    def load_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


if __name__ == "__main__":
    model = VisionModel("Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    model.manual_load()
    res = model.inference(
        sys_prompt="hello",
        user_prompt="respon wassap beigin",
    )
    print(res)
    model.manual_unload()
