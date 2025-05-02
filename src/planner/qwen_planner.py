import re

from models.qwen_vl_model import Qwen2_5VLModel
from planner.base import Plan, PlannerInterface
import logging

from utils.logging_utils import setup_logger, log_exception, log_function_entry_exit

class QwenVLPlanner(PlannerInterface, Qwen2_5VLModel):
    """
    Planner implementation for QwenVL based models. It also supports derivatives such as OSAtlas
    """
    logger: logging.Logger = None

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.logger = setup_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    def plan(self, sys_prompt, user_prompt, *args, **kwargs):
        # Create messages list for context
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Call the model directly
        processed_output_text = self._call(user_prompt, sys_prompt=sys_prompt, *args, **kwargs)

        if not processed_output_text:
            self.logger.error("No output was given by the model")
            raise RuntimeError("Something went wrong while generating the plan and no output was given by the model")

        return self.parse_plan(messages, processed_output_text)

    def parse_plan(self, prompt: list[dict[str, str]], plan: str) -> Plan:
        reasoning_pattern = r"<\|reasoning_begin\|>(.*?)<\|reasoning_end\|>"
        steps_pattern = r"<\|steps_begin\|>(.*?)<\|steps_end\|>"

        reasoning_match = re.search(reasoning_pattern, plan, re.DOTALL)
        steps_match = re.search(steps_pattern, plan, re.DOTALL)

        reasoning_content = reasoning_match.group(1).strip() if reasoning_match else None
        steps_content = steps_match.group(1).strip() if steps_match else None

        if type(steps_content) is not str:
            self.logger.error("No steps were found in the plan for the following response: %s", plan)
            raise RuntimeError("No steps were found in the plan")
        steps: list[str] = list(map(lambda x: x.strip(), steps_content.split(",")))

        if reasoning_content is None:
            self.logger.info("No reasoning was found in the plan")
            return Plan(prompt, plan, steps)

        reasoning_dict = {}
        sections = re.split(r"\n\d+\.\s", reasoning_content)

        for section in sections[1:]:
            lines = section.strip().split("\n- ")
            key = lines[0].strip()
            values = [line.strip() for line in lines[1:]]
            reasoning_dict[key] = values

        return Plan(prompt, plan, steps, reasoning=reasoning_dict)
