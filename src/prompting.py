from action.base import Action, ActionResult, History
from action.qwen_action import AtlasActionmodel, QwenVLActionModel
from planner.base import Plan
from planner.qwen_planner import QwenVLPlanner
from prompts.action_prompts import SYS_PROMPT_ATLASPRO
from prompts.planner_prompts import SYS_PROMPT_COT as PLANNER_COT
from utils.logging_utils import (
    setup_logger,
    log_exception,
    log_function_entry_exit,
    LoggedException,
)

from PIL import Image
import torch
import sys
import logging
import json

import argparse

# Initialize logger
logger = setup_logger(__name__)


@log_function_entry_exit(logger)
def take_action(
    subtask: str,
    history: History,
    image: Image.Image,
    task: str,
    plan: Plan,
    context: str,
    task_description: str,
) -> None:
    """
    Performs an action on the current screen given an instruction

    @param subtask: Subtask at hand from the original plan
    @param history: Actions and results until this point
    @image_path: Path to the image upon which the interaction will happen
    @task: Objective
    @param plan: Plan layed out by the planner beforehand
    @context: Bussiness context
    @task_description: Detailed description of the task at hand, from a process POV
    """
    logger.info(f"Executing subtask: {subtask}")
    logger.debug(f"Current history state: {len(history.actions)} actions performed")

    try:
        middle_model = QwenVLActionModel("Qwen/Qwen2.5-VL-7B-Instruct")
        middle_model.manual_load()
        logger.debug("Initialized QwenVLActionModel")

        prompt = f"""
        **task**: {task}
        **plan**: {plan.steps}.
        **Plan reasoning**: {plan.reasoning}

        **history**:
        {history}

        **result of the last executed action**: {history.last_result}

        **task description**:
        {task_description}

        **context description**:
        {context}

        **current subtask**: {subtask}
        """

        logger.debug("Generating action with middle model")
        action: Action = middle_model.action(
            SYS_PROMPT_ATLASPRO,
            prompt,
            image=image,
        )
        logger.info(
            f"Generated action: {action.action} with target: {action.action_target}"
        )

        logger.debug("Initializing Atlas action model for grounding")
        action_model = AtlasActionmodel("OS-Copilot/OS-Atlas-Base-4B")

        grounding_prompt = f'In this UI screenshot, what is the position of the element corresponding to the command "{action.action_target}" (with bbox)?'
        logger.debug(f"Grounding with prompt: {grounding_prompt}")

        grounding: Action = action_model.action(
            None,
            grounding_prompt,
            image=image,
        )
        grounding.action = action.action
        grounding.action_target = action.action_target

        logger.info(f"Grounded action to coordinates: {grounding.coords}")
        history.append(grounding, ActionResult.PENDING)
        logger.debug("Added action to history with PENDING status")

        middle_model.manual_unload()
        del middle_model
        torch.cuda.empty_cache()

    except Exception as e:
        log_exception(
            logger,
            e,
            {
                "subtask": subtask,
                "task": task,
                "history_length": len(history.actions) if history else 0,
            },
        )
        logger.error(f"Failed to execute subtask: {subtask}")
        raise LoggedException()


@log_function_entry_exit(logger)
def plan_task(
    task: str,
    image: Image.Image,
    context: str,
    task_description: str,
) -> Plan:
    """
    Plans ahead the steps to carry out to complete the given task

    @param task: Task to complete on the user's computer
    @image_path: Path to the image upon which the interaction will happen
    @context: Bussiness context
    @task_description: Detailed description of the task at hand, from a process POV

    @returns plan: Plan object
    """
    logger.info(f"Planning task: {task}")

    try:
        # This prompt now resembles a process description
        planner = QwenVLPlanner("Qwen/Qwen2.5-VL-7B-Instruct")
        planner.manual_load()
        logger.debug("Initialized QwenVLPlanner")

        prompt = f"""
        **Task Description**: {task}.
        **Contextual Information**:
        {task_description}
        """

        logger.debug(f"Generated planning prompt with task: {task}")
        formatted_sys_prompt = PLANNER_COT.format(context=context)

        plan: Plan = planner.plan(
            formatted_sys_prompt,
            prompt,
            image=image,
        )

        logger.info(f"Generated plan with {len(plan.steps)} steps")
        logger.debug(f"Plan steps: {json.dumps(plan.steps, indent=2)}")

        planner.manual_unload()
        del planner
        torch.cuda.empty_cache()

        return plan

    except Exception as e:
        log_exception(
            logger,
            e,
            {
                "task": task,
                "context_length": len(context) if context else 0,
                "task_description_length": len(task_description)
                if task_description
                else 0,
            },
        )
        logger.error(f"Failed to plan task: {task}")
        raise LoggedException()


def parse_args():
    parser = argparse.ArgumentParser(description="Task automation script")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info("Starting application")

        task: str = 'Register a client with email "example@email.com" and password "password123"'
        logger.debug(f"Task defined: {task}")

        context: str = """
        - The organization operates in a legal advisory setting.
        - Users are registered in the Odoo system.
        - Chrome is used as the main browser
        - Gmail is used as the email client
        """
        logger.debug("Context defined")

        task_description: str = """
        - First, check email from user to see if a NIF was sent
        - If it was, register user, else respond to email and stop process
        - Register the user in Odoo
        - Send email back to user confirming registration
        """
        logger.debug("Task description defined")

        history = History()
        logger.debug("History initialized")

        logger.info("Opening image file")
        try:
            image = Image.open("./resources/A.png")
            logger.debug("Image loaded successfully")
        except Exception as e:
            log_exception(logger, e, {"image_path": "./resources/A.png"})
            logger.error(
                "Failed to load image. Make sure the resources directory exists with the correct image file."
            )
            sys.exit(1)

        logger.info("Creating plan")
        plan = plan_task(
            task,
            image=image,
            context=context,
            task_description=task_description,
        )

        logger.info("Executing first step of plan")
        take_action(
            plan.steps[0],
            history,
            image=image,
            task=task,
            plan=plan,
            context=context,
            task_description=task_description,
        )

        print("\n-----------------------\n", history)
        logger.info("Application completed successfully")

    except LoggedException:
        logger.error("Application terminated due to an error in the task execution")
        sys.exit(1)
    except Exception as e:
        # log_exception(logger, e)
        logger.critical("Application terminated with an unhandled exception")
        sys.exit(1)
