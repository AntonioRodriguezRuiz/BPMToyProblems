# The purpose of this file is to write an algoritm that given a broken trace from an rpa robot, the agent is able to solve it and give control back to the robot
# We will have:
# 1. Current state of the robot
# 2. The broken trace
# 3. The expected trace (Original process UI Log)
# 4. Final objective of the agent (next state of the robot)

# The agent is allowed to iteratively act over the screen until one of the following conditions is met:
# 1. The agent solves the problem and gives control back to the robot
# 2. The agent is unable to solve the problem and raises the error to a human operator

import polars as pl
import os
import sys
from PIL import Image
import torch
import logging
import json
import re

from action import Action, ActionResult, History, QwenVLActionModel
from planner import Plan, QwenVLPlanner
from models import TextModel
from utils.logging_utils import (
    setup_logger,
    log_exception,
    log_function_entry_exit,
    LoggedException,
    log_variable,
)
from prompts import (
    RECOVERY_PLANNER_PROMPT,
    RECOVERY_ACTION_PROMPT,
    RECOVERY_PLAN_VALIDATOR_PROMPT,
)

# Initialize logger
logger = setup_logger(__name__)

problems = {
    1: {
        "id": 1,
        "log_path": "resources/ToyProblem1/log.csv",
        "last_successful_action": 2,
        "robot_last_screenshot": "resources/ToyProblem1/images_new/1_2.png",
        "expected_solution": [
            (
                "Click on continue button",
                [(730, 350), (730, 404), (872, 350), (872, 404)],
            ),
            (
                "Click on password input field",
                [(730, 350), (730, 404), (1180, 350), (1180, 404)],
            ),
            ("Type in password", None),
            ("Click on login button", [(730, 427), (730, 484), (854, 427), (854, 484)]),
        ],
        "expected_screenshots": [
            "resources/ToyProblem1/images_new/1_2.png",  # On first click
            "resources/ToyProblem1/images_new/1_2_5.png",  # On first click
            "resources/ToyProblem1/images_new/1_2_5.png",  # On second click
            "resources/ToyProblem1/images_new/1_2_5_2.png",  # After input
            "resources/ToyProblem1/images_new/1_2_5_2.png",  # On last click
        ],
    },
    2: {
        "id": 2,
        "log_path": "resources/ToyProblem2/log.csv",
        "last_successful_action": 3,
        "robot_last_screenshot": "resources/ToyProblem2/images_new/robot_last_screenshot.png",
    },
}


class Problem:
    def __init__(
        self,
        id,
        log_path,
        robot_trace,
        last_successful_action_idx,
        last_successful_action,
        robot_last_screenshot,
        expected_action,
        expected_solution,
    ):
        self.id = id
        self.log_path = log_path
        self.robot_trace = robot_trace
        self.last_successful_action_idx = last_successful_action_idx
        self.last_successful_action = last_successful_action  # Semantic description
        self.robot_last_screenshot = robot_last_screenshot
        self.expected_action = expected_action  # Semantic description
        self.expected_solution: list[tuple[None | str | list[tuple[int]]]] = (
            expected_solution  # List of expected actions to be performed
        )

    def __repr__(self):
        return f"Problem(id={self.id}, log_path={self.log_path}, last_successful_action={self.last_successful_action})"


def get_problem(problem_id):
    """
    Given a problem id, return the problem object

    :param problem_id: The id of the problem
    :return: The problem object
    """
    event_log = pl.read_csv(problems[problem_id]["log_path"])
    robot_trace = event_log[problems[problem_id]["last_successful_action"] :]

    last_successful_action = event_log.row(
        problems[problem_id]["last_successful_action"] - 1, named=True
    )["EventDescription"]
    expected_action = event_log.row(
        problems[problem_id]["last_successful_action"], named=True
    )["EventDescription"]

    problem = Problem(
        id=problem_id,
        log_path=problems[problem_id]["log_path"],
        robot_trace=robot_trace,
        last_successful_action_idx=problems[problem_id]["last_successful_action"],
        last_successful_action=last_successful_action,
        robot_last_screenshot=problems[problem_id]["robot_last_screenshot"],
        expected_action=expected_action,
        expected_solution=problems[problem_id].get("expected_solution", []),
    )

    return problem


@log_function_entry_exit(logger)
def plan_recovery(problem, current_screenshot):
    """
    Creates a plan to recover from a broken RPA trace

    :param problem: The problem object containing information about the broken trace
    :param current_screenshot: Image showing the current state of the application
    :return: A Plan object with steps to recover
    """
    logger.info(f"Planning recovery for problem {problem.id}")

    try:
        planner = QwenVLPlanner("Qwen/Qwen2.5-VL-32B-Instruct")
        planner.manual_load()
        logger.debug("Initialized QwenVLPlanner for recovery")

        # Create prompt with problem-specific context
        prompt = f"""
        **Recovery Task**: Recover from a broken RPA process and get it back on track.
        
        **Last Successful Action**: {problem.last_successful_action}
        
        **Failed Action (Objective)**: {problem.expected_action}
        
        **Process Context**:
        This is part of a {problem.robot_trace[0]["ActivityLabel"]} workflow.
        The robot was performing a sequence of UI interactions when it failed.
        
        **Goal**:
        Identify what went wrong and provide steps to recover the process so the robot can continue.
        """

        logger.debug(f"Generated recovery planning prompt for problem {problem.id}")

        # Get the recovery plan
        plan = planner.plan(RECOVERY_PLANNER_PROMPT, prompt, image=current_screenshot)

        logger.info(f"Generated recovery plan with {len(plan.steps)} steps")
        logger.debug(f"Recovery plan steps: {json.dumps(plan.steps, indent=2)}")

        # Log the generated plan to variables log file
        log_variable(
            "recovery_plan",
            {
                "problem_id": problem.id,
                "steps": plan.steps,
                "reasoning": plan.reasoning,
            },
            {"prompt": prompt},
        )

        planner.manual_unload()
        del planner
        torch.cuda.empty_cache()

        # Validate the plan against expected solutions
        score = validate_recovery_plan(problem, plan.steps, current_screenshot)
        if score != "Pass":
            logger.warning(f"Recovery plan validation failed with score: {score}")
            # Log the validation failure
            log_variable(
                "plan_validation_failure",
                {"problem_id": problem.id, "score": score, "plan": plan.steps},
                {"expected_solution": problem.expected_solution},
            )
            raise LoggedException("Recovery plan validation failed")

        return plan

    except Exception as e:
        log_exception(
            logger,
            e,
            {
                "problem_id": problem.id,
                "last_successful_action": problem.last_successful_action,
                "expected_action": problem.expected_action,
            },
        )
        logger.error(f"Failed to plan recovery for problem {problem.id}")
        raise LoggedException()


@log_function_entry_exit(logger)
def validate_recovery_plan(problem, plan, current_screenshot):
    """
    Validates a recovery plan against the expected solution for a problem

    :param problem: The problem object containing information about expected solutions
    :param plan: The generated recovery plan to validate
    :param current_screenshot: Current screenshot of the application
    :return: Validation score (Pass/Fail/Partial)
    """
    logger.info(f"Validating recovery plan for problem {problem.id}")

    validator = TextModel("Qwen/Qwen2.5-32B-Instruct")
    expected_steps = list(map(lambda x: x[0], problem.expected_solution))
    prompt = f"""
    **Plan Validation Task**: Validate the recovery plan for a broken RPA process.
    **Last Successful Action**: {problem.last_successful_action}
    **Failed Action (Objective)**: {problem.expected_action}
    **Recovery Plan**: {json.dumps(plan, indent=2)}
    **Expected Solution**: {json.dumps(expected_steps, indent=2)}
    """

    validator.manual_load()
    validation_result = validator(
        prompt, sys_prompt=RECOVERY_PLAN_VALIDATOR_PROMPT, image=current_screenshot
    )
    logger.info(f"Generated plan validation for problem {problem.id}")
    validator.manual_unload()
    del validator
    torch.cuda.empty_cache()

    # The result is wthin a JSON object in the response
    json_pattern = r"```json\s*(.*?)\s*```"
    json_match = re.search(json_pattern, validation_result, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        # If not found between code blocks, try to find a JSON object directly
        json_pattern = r"\{.*?\}"
        json_match = re.search(json_pattern, validation_result, re.DOTALL)
        if json_match:
            json_str = json_match.roup(0)
        else:
            logger.error("No JSON content found in the validator response")

    try:
        plan_data = json.loads(json_str)
        log_variable(
            "plan_validation",
            {"problem_id": problem.id, "plan": plan, "validation_result": plan_data},
            {"expected_solution": expected_steps},
        )
        score = plan_data.get("score", {})

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from validation: {e}")

    return score


@log_function_entry_exit(logger)
def execute_recovery_step(step, history, current_screenshot, problem, plan):
    """
    Executes a single step in the recovery plan

    :param step: The current step to execute
    :param history: History of actions taken during recovery
    :param current_screenshot: The current screenshot of the application
    :param problem: The problem object
    :param plan: The recovery plan
    :return: ActionResult indicating the result of the execution
    """
    logger.info(f"Executing recovery step: {step}")
    logger.debug(f"Current history state: {len(history.actions)} actions performed")

    try:
        action_model = QwenVLActionModel("Qwen/Qwen2.5-VL-7B-Instruct")
        action_model.manual_load()
        logger.debug("Initialized QwenVLActionModel")

        # Create prompt with problem-specific context
        prompt = f"""
        **Recovery Task**: Recover from a broken RPA process and get it back on track.
        
        **Last Successful Action**: {problem.last_successful_action}
        
        **Expected Action (Failed)**: {problem.expected_action}
        
        **Recovery Plan**: {", ".join(plan.steps)}.
        
        **Plan Reasoning**: {plan.reasoning}
        
        **History**:
        {history}
        
        **Result of Last Action**: {history.last_result}
        
        **Current Step to Execute**: {step}
        """

        # Generate the action
        action = action_model.action(
            RECOVERY_ACTION_PROMPT, prompt, image=current_screenshot
        )

        logger.info(
            f"Generated action: {action.action} with target: {action.action_target}"
        )

        # Log the action to the variables log file
        log_variable(
            "action_execution",
            {
                "step": step,
                "action_type": action.action,
                "action_target": action.action_target,
                "problem_id": problem.id,
            },
            {
                "plan_context": plan.steps,
                "history_length": len(history.actions) if history else 0,
            },
        )

        # TODO: Check coordinates for actions that involve clicking.
        result = ActionResult.SUCCESS

        # Add action to history
        history.append(action, result)
        logger.debug("Added action to history with SUCCESS status")

        action_model.manual_unload()
        del action_model
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        # Log the failed action to variables log file
        log_variable(
            "action_execution_failure",
            {
                "step": step,
                "problem_id": problem.id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            {
                "history_length": len(history.actions) if history else 0,
                "plan_context": plan.steps,
            },
        )

        log_exception(
            logger,
            e,
            {
                "step": step,
                "problem_id": problem.id,
                "history_length": len(history.actions) if history else 0,
            },
        )
        logger.error(f"Failed to execute recovery step: {step}")
        raise LoggedException()


def solve_problem(problem_id, max_retry_attempts=3):
    """
    Main function to solve a toy problem

    :param problem_id: The ID of the problem to solve
    :param max_retry_attempts: Maximum number of retry attempts for recovery
    :return: True if the problem was solved, False otherwise
    """
    logger.info(f"Starting to solve problem {problem_id}")

    try:
        # Get problem definition
        problem = get_problem(problem_id)
        logger.info(f"Retrieved problem: {problem}")

        image_path = os.path.join(
            os.path.dirname(problem.log_path), problem.robot_trace[0]["Screenshot"][0]
        )

        logger.info(f"Loading screenshot from {image_path}")
        try:
            current_screenshot = Image.open(problem.robot_last_screenshot)
            logger.debug("Screenshot loaded successfully")
        except Exception as e:
            log_exception(logger, e, {"image_path": image_path})
            # Log the validation failure for screenshot loading
            log_variable(
                "validation_failure",
                {
                    "component": "screenshot_loading",
                    "problem_id": problem_id,
                    "image_path": problem.robot_last_screenshot,
                    "error": str(e),
                },
            )
            logger.error("Failed to load screenshot. Make sure the path is correct.")
            return False

        # Initialize history
        history = History()
        logger.debug("History initialized")

        # Create recovery plan
        logger.info("Creating recovery plan")
        plan = plan_recovery(problem, current_screenshot)

        # Execute recovery steps
        logger.info("Executing recovery steps")
        attempt = 0

        while attempt < max_retry_attempts:
            logger.info(f"Recovery attempt {attempt + 1} of {max_retry_attempts}")

            success = True
            for i, step in enumerate(plan.steps):
                logger.info(f"Executing step {i + 1} of {len(plan.steps)}: {step}")

                result = execute_recovery_step(
                    step, history, current_screenshot, problem, plan
                )

                if result != ActionResult.SUCCESS:
                    logger.warning(f"Step {i + 1} failed with result {result}")
                    # Log the step execution failure
                    log_variable(
                        "step_validation_failure",
                        {
                            "problem_id": problem_id,
                            "attempt": attempt + 1,
                            "step_number": i + 1,
                            "step": step,
                            "result": str(result),
                        },
                        {
                            "plan_context": plan.steps,
                            "history": history.get_formatted_history(),
                        },
                    )
                    success = False
                    break

                # In a real implementation, we would update the screenshot here
                # For this toy problem, we'll simulate moving to the next screenshot
                next_idx = min(
                    problem.last_successful_action_idx + i + 1,
                    len(problem.robot_trace) - 1,
                )
                if next_idx < len(problem.robot_trace):
                    next_image_path = os.path.join(
                        os.path.dirname(problem.log_path),
                        problem.robot_trace[next_idx]["Screenshot"][0],
                    )
                    try:
                        current_screenshot = Image.open(next_image_path)
                        logger.debug(f"Updated screenshot to {next_image_path}")
                    except Exception as e:
                        log_exception(logger, e, {"image_path": next_image_path})
                        # Log screenshot transition failure
                        log_variable(
                            "screenshot_transition_failure",
                            {
                                "problem_id": problem_id,
                                "attempt": attempt + 1,
                                "step_number": i + 1,
                                "current_image": problem.robot_last_screenshot,
                                "next_image_path": next_image_path,
                                "error": str(e),
                            },
                        )
                        logger.warning(
                            f"Could not load next screenshot {next_image_path}, continuing with current"
                        )
                        raise LoggedException()

            if success:
                # Log successful recovery
                log_variable(
                    "recovery_success",
                    {
                        "problem_id": problem_id,
                        "attempts_needed": attempt + 1,
                        "steps_executed": len(plan.steps),
                        "total_actions": len(history.actions),
                    },
                    {"plan": plan.steps, "reasoning": plan.reasoning},
                )
                logger.info("Recovery successful!")
                return True

            # If we get here, recovery failed - try again with updated plan
            logger.warning(f"Recovery attempt {attempt + 1} failed, retrying")
            # Log failed recovery attempt
            log_variable(
                "recovery_attempt_failure",
                {
                    "problem_id": problem_id,
                    "attempt_number": attempt + 1,
                    "steps_executed": i + 1,
                    "total_actions": len(history.actions),
                },
                {"plan": plan.steps, "history": history.get_formatted_history()},
            )
            plan = plan_recovery(problem, current_screenshot)
            attempt += 1

        # Log maximum retries reached
        log_variable(
            "max_retries_reached",
            {
                "problem_id": problem_id,
                "max_retry_attempts": max_retry_attempts,
                "total_actions": len(history.actions),
            },
            {"last_plan": plan.steps, "history": history.get_formatted_history()},
        )
        logger.error(f"Failed to recover after {max_retry_attempts} attempts")
        return False

    except LoggedException:
        logger.error("Problem solving terminated due to a logged exception")
        return False
    except Exception as e:
        log_exception(logger, e, {"problem_id": problem_id})
        logger.critical("Problem solving terminated with an unhandled exception")
        return False


def main():
    """
    Main entry point for solving toy problems
    """
    import argparse

    parser = argparse.ArgumentParser(description="Solve RPA toy problems")
    parser.add_argument(
        "--problem_id", type=int, help="ID of the problem to solve", default=1
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging", default=False
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum number of retry attempts"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.problem_id not in problems:
        logger.error(f"Problem ID {args.problem_id} not found")
        return 1

    success = solve_problem(args.problem_id, args.max_retries)

    if success:
        logger.info(f"Problem {args.problem_id} solved successfully")
        return 0
    else:
        logger.error(f"Failed to solve problem {args.problem_id}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
