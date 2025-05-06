import re
import json
from typing import Tuple

import jax.numpy as jnp

from action.base import Action, ActionInterface
from models.qwen_vl_model import Qwen2_5VLModel


class QwenVLActionModel(ActionInterface, Qwen2_5VLModel):
    """
    ActionModel implentation for QwenVL based models. It also supports derivatives such as OSAtlas
    """

    capabilities: list[str] = ["image", "text"]

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)

    def action(self, sys_prompt, user_prompt, *args, **kwargs) -> Action:
        # Call the inference method directly instead of _call
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Process the model's output
        processed_output_text = self._call(
            user_prompt, sys_prompt=sys_prompt, *args, **kwargs
        )

        if not processed_output_text:
            raise RuntimeError(
                "Something went wrong while generating the action and no output was given by the model"
            )

        return self.parse_action(messages, processed_output_text)

    def parse_action(self, prompt: list[dict[str, str]], model_response: str) -> Action:
        """
        Parse JSON-formatted action response

        :param prompt: The original prompt given to the model
        :param model_response: The model's JSON response
        :return: An Action object with parsed action details
        """
        # Extract JSON content from the response
        json_pattern = r"```json\s*(.*?)\s*```"
        json_match = re.search(json_pattern, model_response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # If not found between code blocks, try to find a JSON object directly
            json_pattern = r"\{.*\}"
            json_match = re.search(json_pattern, model_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fall back to the old regex pattern for backward compatibility
                return self._parse_action_with_regex(prompt, model_response)

        try:
            action_data = json.loads(json_str)

            # Extract action details from the JSON
            context_analysis = action_data.get("context_analysis", "")

            action_info = action_data.get("action", {})
            action_type = action_info.get("type", None)
            action_target = action_info.get("target", None)

            if not action_type:
                # Fall back to the old regex pattern for backward compatibility
                return self._parse_action_with_regex(prompt, model_response)

            return Action(
                prompt,
                action_target,
                model_response,
                action=action_type,
                reasoning=context_analysis,
            )

        except json.JSONDecodeError:
            # Fall back to the old regex pattern for backward compatibility
            return self._parse_action_with_regex(prompt, model_response)

    def _parse_action_with_regex(
        self, prompt: list[dict[str, str]], model_response: str
    ) -> Action:
        """
        Legacy method to parse action using regex patterns for backward compatibility
        """
        reasoning_pattern = (
            r"<\|context_analysis_begin\|>(.*?)<\|context_analysis_end\|>"
        )
        action_name_pattern = r"<\|action_begin\|>(.*?)<\|action_end\|>"
        action_target_pattern = r"\[(.*)\]"

        reasoning_match = re.search(reasoning_pattern, model_response, re.DOTALL)
        action_name_match = re.search(action_name_pattern, model_response, re.DOTALL)
        action_target_match = re.search(
            action_target_pattern, model_response, re.DOTALL
        )

        reasoning_content = (
            reasoning_match.group(1).strip() if reasoning_match else None
        )
        action_name_content = (
            action_name_match.group(1).strip() if action_name_match else None
        )
        action_target_content: str | None = (
            action_target_match.group(1).strip() if action_target_match else None
        )

        if reasoning_content is None:
            return Action(
                prompt, action_target_content, model_response, action_name_content
            )

        reasoning = re.split(r"\n\d+\.\s", reasoning_content)

        return Action(
            prompt,
            action_target_content,
            model_response,
            action=action_name_content,
            reasoning=reasoning,
        )


class AtlasActionModel(QwenVLActionModel):
    """
    ActionModel implementation for Atlas models.
    """

    def parse_action(self, prompt: list[dict[str, str]], model_response: str):
        """
        Parse Atlas model response, which may be JSON or may use the object_ref format
        """
        # First try parsing as JSON
        json_pattern = r"```json\s*(.*?)\s*```"
        json_match = re.search(json_pattern, model_response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                action_data = json.loads(json_str)

                # Extract action details from the JSON
                action_info = action_data.get("action", {})
                action_type = action_info.get("type", None)
                action_target = action_info.get("target", None)

                # If bbox coordinates are included in the JSON
                bbox = action_data.get("bbox", None)
                coords = None

                if bbox and isinstance(bbox, list) and len(bbox) > 0:
                    # Calculate center of bounding box if it's provided
                    element_bbox = jnp.asarray(bbox)
                    coords = tuple(jnp.mean(element_bbox, axis=0).tolist())

                return Action(
                    prompt,
                    action_target,
                    model_response,
                    action=action_type,
                    coords=coords,
                )

            except json.JSONDecodeError:
                # Fall back to Atlas-specific format
                pass

        # If JSON parsing fails, use Atlas-specific format parsing
        object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

        object_ref_match = re.search(object_ref_pattern, model_response, re.DOTALL)
        box_match = re.search(box_pattern, model_response, re.DOTALL)

        object_ref_content = (
            object_ref_match.group(1).strip() if object_ref_match else None
        )
        box_content = box_match.group(1).strip() if box_match else None
        coords: Tuple[float, float]
        if box_content:
            num_pattern = r"(\d+).*?(\d+)"  # Number then closest number to it
            nums = re.findall(num_pattern, box_content)
            element_bbox = jnp.asarray([(int(x), int(y)) for x, y in nums])
            coords = tuple(jnp.mean(element_bbox, axis=0).tolist())

        return Action(prompt, object_ref_content, model_response, coords=coords)
