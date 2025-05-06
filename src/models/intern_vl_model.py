from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Process, Queue

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

from models.vision_model import VisionModel
from models.utils import load_image
import utils.images as images_utils


class InternVLModel(VisionModel):
    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
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
            question = ""
            if sys_prompt:
                question += sys_prompt + "\n"
            question += user_prompt
            generation_config = dict(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=151645,
            )
            pixel_values = None
            if "image" in kwargs:
                question = f"<image>\n{question}"

                # Wide images are badly processed. Thus, we are going to make sure the image has an aspect ratio of 16:9, with a resolution of 1920 x 1080
                # But to not stretch the image, we will do it by adding black padding and putting the original image at the center
                image = kwargs["image"]
                width, height = image.size
                aspect_ratio = width / height
                if aspect_ratio < 16 / 9:
                    new_width = int(height * 16 / 9)
                    padding = (new_width - width) // 2
                    image = images_utils.add_padding(image, padding, 0)
                elif aspect_ratio > 16 / 9:
                    new_height = int(width * 9 / 16)
                    padding = (new_height - height) // 2
                    image = images_utils.add_padding(image, 0, padding)
                image = image.resize((1920, 1080))

                pixel_values = (
                    load_image(image, max_num=12)
                    .to(torch.bfloat16 if "Intern" in self.model_name else torch.int8)
                    .cuda()
                )
            processed_output_text = self._local_inference(
                question, generation_config, pixel_values
            )
        if "result_queue" in kwargs:
            kwargs["result_queue"].put(processed_output_text)
            return None
        else:
            return processed_output_text

    def _local_inference(  # type: ignore[override]
        self,
        question: str,
        generation_config: Dict[str, Any],
        pixel_values=None,
    ) -> str:
        if self.loaded:
            model: AutoModel
            tokenizer: AutoTokenizer
            model, tokenizer = self._loaded_model, self._loaded_processor
        else:
            model, tokenizer = self.load_model()

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
        )

        self.unload(model, tokenizer, pixel_values, question)
        return response

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
        max_tokens: int = kwargs.pop("max_tokens", 1024)
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

    def load_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = (
            AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer
