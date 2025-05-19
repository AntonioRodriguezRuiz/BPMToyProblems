# The purpose of this file is to write an algoritm that given a broken trace from an rpa robot, the agent is able to solve it and give control back to the robot
# We will have:
# 1. Current state of the robot
# 2. The broken trace
# 3. The expected trace (Original process UI Log)
# 4. Final objective of the agent (next state of the robot)

# The agent is allowed to iteratively act over the screen until one of the following conditions is met:
# 1. The agent solves the problem and gives control back to the robot
# 2. The agent is unable to solve the problem and raises the error to a human operator
import logging
from utils.logging_utils import (
    setup_logger,
    log_exception,
    log_function_entry_exit,
    LoggedException,
    log_variable,
)

# Initialize logger
logger = setup_logger(__name__)
logger.propagate = False

import polars as pl
import sys
from PIL import Image
import torch
import json
import re
from matplotlib import pyplot as plt

from action import (
    Action,
    ActionResult,
    History,
    UITarsActionModel,
    QwenOmniparserActionModel,
)
from planner import Plan, QwenVLPlanner
from models import TextModel, Qwen2_5VLModel
from prompts import (
    SYS_PROMPT_MID,
    RECOVERY_PLANNER_PROMPT,
    RECOVERY_PLAN_VALIDATOR_PROMPT,
    SYS_PROMPT_ATLASPRO,
    UITARS_GROUNDING,
    SYS_PROMPT_OMNIPARSER,
)
from utils import resize_img


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
            "resources/ToyProblem1/images_new/1_2.png",  # On wait
            "resources/ToyProblem1/images_new/1_2_5.png",  # On second click
            "resources/ToyProblem1/images_new/1_2_5.png",  # On input
            "resources/ToyProblem1/images_new/1_2_5_2.png",  # On last click
        ],
        "error": 'UI Component for "password input" not found',
        "focused_window": "Firefox Browser",
        "variables": {
            "username": "username",
            "password": "Hs629$hb",
            "clipboard_content": None,
        },
    },
    2: {
        "id": 2,
        "log_path": "resources/ToyProblem2/log.csv",
        "last_successful_action": 30,
        "robot_last_screenshot": "resources/ToyProblem2/images_new/3_1.png",
        "expected_solution": [
            (
                "Any action is valid in this step",
                [(0, 0), (0, 1080), (1920, 0), (1920, 1080)],
            )
            * 15
        ],
        "expected_screenshots": [
            "resources/ToyProblem2/images_new/3_1.png",  # On click
            "resources/ToyProblem2/images_new/3_1.png",  # On click
            "resources/ToyProblem2/images_new/3_1.png",  # On click
            "resources/ToyProblem2/images_new/3_2.png",  # On type
            "resources/ToyProblem2/images_new/3_3.png",  # On click
            "resources/ToyProblem2/images_new/3_4.png",  # On type
            "resources/ToyProblem2/images_new/3_5.png",  # On click
            "resources/ToyProblem2/images_new/3_6.png",  # On type
            "resources/ToyProblem2/images_new/3_7.png",  # On click
            "resources/ToyProblem2/images_new/3_8.png",  # On type
            "resources/ToyProblem2/images_new/3_9.png",  # On click
            "resources/ToyProblem2/images_new/3_10.png",  # On type
            "resources/ToyProblem2/images_new/3_11.png",  # On click
            "resources/ToyProblem2/images_new/3_12.png",  # On type
            "resources/ToyProblem2/images_new/3_13.png",  # On click
            "resources/ToyProblem2/images_new/3_14.png",  # On type
            "resources/ToyProblem2/images_new/3_15.png",  # On submit
        ],
        "error": 'UI Component for "First Name" not found',
        "focused_window": "Firefox Browser",
        "variables": {
            "clipboard_content": "Albert",
        },
    },
}


class Problem:
    def __init__(
        self,
        id,
        log_path,
        robot_trace,
        error,
        focused_window,
        last_successful_action_idx,
        last_successful_action,
        robot_last_screenshot,
        expected_action,
        expected_solution,
        expected_screenshots,
        variables,
    ):
        self.id = id
        self.log_path = log_path
        self.robot_trace = robot_trace
        self.error = error
        self.focused_window = focused_window
        self.last_successful_action_idx = last_successful_action_idx
        self.last_successful_action = last_successful_action  # Semantic description
        self.robot_last_screenshot = robot_last_screenshot
        self.expected_action = expected_action  # Semantic description
        self.expected_solution: list[tuple[None | str | list[tuple[int]]]] = (
            expected_solution  # List of expected actions to be performed
        )
        self.expected_screenshots = expected_screenshots
        self.variables = variables  # Variables to be used in the process

    def __repr__(self):
        return f"Problem(id={self.id}, log_path={self.log_path}, last_successful_action={self.last_successful_action})"


def get_problem(problem_id):
    """
    Given a problem id, return the problem object

    :param problem_id: The id of the problem
    :return: The problem object
    """
    event_log = pl.read_csv(problems[problem_id]["log_path"])
    robot_trace = event_log[: problems[problem_id]["last_successful_action"]]

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
        error=problems[problem_id]["error"],
        focused_window=problems[problem_id]["focused_window"],
        last_successful_action_idx=problems[problem_id]["last_successful_action"],
        last_successful_action=last_successful_action,
        robot_last_screenshot=problems[problem_id]["robot_last_screenshot"],
        expected_action=expected_action,
        expected_solution=problems[problem_id].get("expected_solution", []),
        expected_screenshots=problems[problem_id].get("expected_screenshots", []),
        variables=problems[problem_id].get("variables", {}),
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
        planner = QwenVLPlanner("Qwen/Qwen2.5-VL-72B-Instruct")
        logger.debug("Initialized QwenVLPlanner for recovery")

        # Create prompt with problem-specific context
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_cols=-1,
            tbl_rows=-1,
        ):
            robot_trace: str = problem.robot_trace.__str__()
            log_variable(
                "robot_trace",
                robot_trace,
            )
        prompt = f"""
        **Recovery Task**: You need to finish the activity: {problem.robot_trace[0]["ActivityLabel"]}.
        
        **Last Successful Action**: {problem.last_successful_action}
        
        **Failed Action**: {problem.expected_action}
        **Error**: {problem.error}
        **Focused Window**: {problem.focused_window}

        **Variables**:
        {problem.variables}
        
        **Process Context**:
        This is part of a {problem.robot_trace[0]["ActivityLabel"]} workflow.
        The robot was performing a sequence of UI interactions when it failed.
        The robot cannot take control back until the current task is fully completed.
        The sequence of previous actions is as follows:
        {robot_trace}
        
        **Goal**:
        Identify what went wrong and provide steps to recover the process by finishing the current task so the robot can continue with the next task.
        """

        logger.debug(f"Generated recovery planning prompt for problem {problem.id}")

        # Get the recovery plan
        plan: Plan = planner.plan(
            RECOVERY_PLANNER_PROMPT, prompt, image=current_screenshot
        )
        del planner
        torch.cuda.empty_cache()

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

        # Validate the plan against expected solutions
        score, coordinate_mapping = validate_recovery_plan(
            problem, plan.steps, current_screenshot
        )
        if score != "Pass":
            logger.warning(f"Recovery plan validation failed with score: {score}")
            # Log the validation failure
            log_variable(
                "plan_validation_failure",
                {"problem_id": problem.id, "score": score, "plan": plan.steps},
                {"expected_solution": problem.expected_solution},
            )
            raise LoggedException("Recovery plan validation failed")

        return plan, coordinate_mapping

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

    prompt = f"""
    **Plan Validation Task**: Validate the recovery plan for a broken RPA process.
    **Last Successful Action**: {problem.last_successful_action}
    **Failed Action (Objective)**: {problem.expected_action}
    **Recovery Plan**: {json.dumps(plan, indent=2)}
    **Expected Solution**: {json.dumps(problem.expected_solution, indent=2)}
    """

    validation_result = validator(
        prompt, sys_prompt=RECOVERY_PLAN_VALIDATOR_PROMPT, image=current_screenshot
    )
    logger.info(f"Generated plan validation for problem {problem.id}")
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
            json_str = json_match.group(0)
        else:
            logger.error("No JSON content found in the validator response")

    try:
        logger.debug(f"Validation result JSON: {json_str}")
        plan_data = json.loads(json_str)
        log_variable(
            "plan_validation",
            {"problem_id": problem.id, "plan": plan, "validation_result": plan_data},
            {"expected_solution": problem.expected_solution},
        )
        score = plan_data.get("score", {})
        coordinate_mapping = plan_data.get("coordinate_mapping", {})

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from validation: {e}")
        log_variable("validation_result", validation_result)

    return score, coordinate_mapping


def infer_step_action(step, history, current_screenshot, problem, plan):
    """
    Infers the action type from the step description

    :param step: The current step to infer action for
    :param history: History of actions taken during recovery
    :param current_screenshot: The current screenshot of the application
    :param problem: The problem object
    :param plan: The recovery plan
    :return: Action type, target, and command
    """
    action_model = Qwen2_5VLModel("Qwen/Qwen2.5-VL-32B-Instruct")
    action_model.manual_load()

    prompt = f"""
    **Task**: {problem.robot_trace[0]["ActivityLabel"]}
    **Plan**: {", ".join(plan.steps)}.
    **Plan Reasoning**: {plan.reasoning}

    **History**:
    {history}

    **Last Action**: {history.last_action or problem.last_successful_action}
    **Result of Last Action**: {ActionResult.SUCCESS}
    
    **Current Subtask**: {step}
    """

    response: Action = action_model(
        prompt, sys_prompt=SYS_PROMPT_MID, image=current_screenshot
    )
    action_model.manual_unload()
    del action_model
    torch.cuda.empty_cache()

    json_pattern = r"```json\s*(.*?)\s*```"
    json_match = re.search(json_pattern, response, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        # If not found between code blocks, try to find a JSON object directly
        json_pattern = r"\{.*?\}"
        json_match = re.search(json_pattern, response, re.DOTALL)
        if json_match:
            json_str = json_match.roup(0)
        else:
            logger.error("No JSON content found in the validator response")

    try:
        plan_data = json.loads(json_str)
        action_data = plan_data.get("action", {})
        log_variable(
            "action_inference",
            {"step": step, "action": response},
        )
        action_type = action_data.get("type", {})
        target = action_data.get("target_id", {})
        command = action_data.get("command", {})

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from validation: {e}")
        log_variable("action_inference_model_response", response)

    return action_type, target, command


@log_function_entry_exit(logger)
def execute_recovery_step(
    action_model, step, gt_coords, history, current_screenshot, problem, plan
):
    """
    Executes a single step in the recovery plan

    :param action_model: The action model to use for execution
    :param step: The current step to execute
    :param gt_coords: Ground truth coordinates for the action (if applicable)
    :param history: History of actions taken during recovery
    :param current_screenshot: The current screenshot of the application
    :param problem: The problem object
    :param plan: The recovery plan
    :return: ActionResult indicating the result of the execution
    """
    logger.info(f"Executing recovery step: {step}")
    logger.debug(f"Current history state: {len(history.actions)} actions performed")

    try:
        # action_type, target, command = infer_step_action(
        #     step, history, current_screenshot, problem, plan
        # )

        prompt = f"""
        **Task**: {problem.robot_trace[0]["ActivityLabel"][0]}
        **Plan**: {".\n ".join(plan.steps)}.
        **Plan Reasoning**: {plan.reasoning}

        **History**:
        {history}

        **Last Action**: {history.last_action or problem.last_successful_action}
        **Result of Last Action**: {ActionResult.SUCCESS}
        
        **Current Subtask (Instruction)**: {step}

        **List of elements:**
        """

        action = action_model.action(
            SYS_PROMPT_OMNIPARSER,
            prompt,
            image=current_screenshot,
        )

        result = ActionResult.SUCCESS  # Default to success if its not a click
        logger.info(f"Generated action: {action.to_str_extended()}")
        log_variable(
            "action_execution",
            {
                "step": step,
                "action_type": action.action,
                "action_target": action.action_target,
                "action_coords": action.coords,
                "raw": action.raw,
                "problem_id": problem.id,
            },
            {
                "plan_context": plan.steps,
                "history_length": len(history.actions) if history else 0,
            },
        )
        if "click" in action.action.lower():
            # grounding_model = UITarsActionModel("ByteDance-Seed/UI-TARS-7B-DPO")
            # grounding_model.manual_load()

            # prompt = f'In this UI screenshot, what is the position of the element corresponding to the command: "{command}" (with bbox)?'

            # action: Action = grounding_model.action(
            #     UITARS_GROUNDING, prompt, image=current_screenshot
            # )

            # grounding_model.manual_unload()
            # del grounding_model
            # torch.cuda.empty_cache()

            ## MOCK GT_COORDS
            gt_coords = [(0, 0), (0, 1080), (1920, 0), (1920, 1080)]

            if gt_coords is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    # Show the coordinates on the image
                    plt.imshow(current_screenshot)
                    plt.scatter(
                        [float(action.coords[0])],
                        [float(action.coords[1])],
                        color="red",
                        label="Action Coordinates",
                    )
                    plt.scatter(
                        [float(gt_coords[0][0]), float(gt_coords[2][0])],
                        [float(gt_coords[0][1]), float(gt_coords[1][1])],
                        color="blue",
                        label="Ground Truth Coordinates",
                    )
                    plt.legend()
                    plt.title(f"Action Coordinates vs Ground Truth for {step}")
                    plt.show()

                # Check if the action coordinates are within the ground truth coordinates
                if float(
                    float(gt_coords[0][0])
                    <= float(action.coords[0])
                    <= float(gt_coords[2][0])
                ) and float(gt_coords[0][1]) <= float(action.coords[1]) <= float(
                    float(gt_coords[1][1])
                ):
                    logger.info(
                        "Action coordinates are within the ground truth coordinates"
                    )
                else:
                    logger.warning(
                        "Action coordinates are outside the ground truth coordinates"
                    )
                    result = ActionResult.FAIL

        history.append(action, result)
        logger.debug("Added action to history with SUCCESS status")

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

        try:
            current_screenshot = Image.open(problem.robot_last_screenshot)
            logger.info(f"Loading screenshot from {problem.robot_last_screenshot}")
            logger.debug("Screenshot loaded successfully")
        except Exception as e:
            log_exception(logger, e, {"image_path": problem.robot_last_screenshot})
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
        plan, coordinate_mapping = plan_recovery(problem, current_screenshot)

        # Execute recovery steps
        logger.info("Executing recovery steps")
        attempt = 0

        success = True
        action_model = QwenOmniparserActionModel("Qwen/Qwen2.5-VL-72B-Instruct")
        action_model.manual_load()
        for i, step in enumerate(plan.steps):
            logger.info(f"Executing step {i + 1} of {len(plan.steps)}: {step}")

            current_screenshot = Image.open(problem.expected_screenshots[i])

            gt_coords = coordinate_mapping.get(str(i + 1), None)

            result = execute_recovery_step(
                action_model,
                step,
                gt_coords,
                history,
                current_screenshot,
                problem,
                plan,
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
                        "history": history,
                    },
                )
                success = False
        action_model.manual_unload()
        del action_model
        torch.cuda.empty_cache()

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
                {"plan": plan.steps, "reasoning": plan.reasoning, "history": history},
            )
            logger.info("Recovery successful!")

        else:
            log_variable(
                "recovery_attempt_failure",
                {
                    "problem_id": problem_id,
                    "steps_executed": i + 1,
                    "total_actions": len(history.actions),
                },
                {"plan": plan.steps, "history": history},
            )
        return success

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
