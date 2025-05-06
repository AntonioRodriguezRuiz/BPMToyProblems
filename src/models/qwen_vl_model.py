import torch
from typing import Any, List, Tuple
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from models.vision_model import VisionModel


class QwenVLModel(VisionModel):
    """
    Implementation of the QwenVL model interface.

    **Example Usage (Local Inference):**

    ```python
    from models.models import QwenVLModel

    # Using a locally loaded QwenVL model
    model = QwenVLModel("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    image = Image.open("path/to/image.jpg")
    output = model(
        "Provide insights on the image", sys_prompt="Image analysis", image=image
    )
    print(output)
    ```

    **Example Usage (Using OpenAI API):**

    ```python
    from models.models import QwenVLModel

    # Configure for OpenAI API inference
    model = QwenVLModel("your-qwenvl-model-id")
    model.openai_server = "https://api.openai.com/v1"
    model.api_key = "YOUR_API_KEY"
    image = Image.open("path/to/image.jpg")
    output = model(
        "Provide insights on the image", sys_prompt="Image analysis", image=image
    )
    print(output)
    ```
    """

    capabilities: List[str] = ["image", "text"]

    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, *args, **kwargs)

    def load_model(self) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]: The loaded model and processor.
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cuda",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
            if "QVQ" in self.model_name
            else None,
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


class Qwen2_5VLModel(QwenVLModel):
    def load_model(self) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
        """
        Loads and returns the model to make inferences on.

        Returns:
            Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]: The loaded model and processor.
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "AWQ" in self.model_name else "auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


if __name__ == "__main__":
    model = QwenVLModel("Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    model.manual_load()
    res = model.inference(
        sys_prompt="hello",
        user_prompt="respon wassap beigin",
    )
    print(res)
    model.manual_unload()
    del model
    torch.cuda.empty_cache()
