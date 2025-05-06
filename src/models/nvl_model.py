import torch
from typing import Any, Tuple
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from models.intern_vl_model import InternVLModel


class NVLModel(InternVLModel):
    def load_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Loads and returns the vision+text model and processor.
        """
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer
