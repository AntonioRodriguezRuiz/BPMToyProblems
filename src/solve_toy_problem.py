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

from action.base import Action, ActionResult, History
from action.qwen_action import QwenVLActionModel
from planner.base import Plan
from planner.qwen_planner import QwenVLPlanner
from utils.logging_utils import setup_logger, log_exception, log_function_entry_exit, LoggedException

# Initialize logger
logger = setup_logger(__name__)

problems = {
    1: {
        "id": 1,
        "log_path": "resources/ToyProblem1/log.csv",
        "last_successful_action": 2,
        "robot_last_screenshot": "resources/ToyProblem1/images_new/1_2.png",
    },
    2: {
        "id": 2,
        "log_path": "resources/ToyProblem2/log.csv",
        "last_successful_action": 3,
        "robot_last_screenshot": "resources/ToyProblem2/images_new/robot_last_screenshot.png",
    }
}

class Problem:
    def __init__(self, id, log_path, robot_trace, last_successful_action_idx, last_successful_action, robot_last_screenshot, expected_action):
        self.id = id
        self.log_path = log_path
        self.robot_trace = robot_trace
        self.last_successful_action_idx = last_successful_action_idx
        self.last_successful_action = last_successful_action # Semantic description
        self.robot_last_screenshot = robot_last_screenshot
        self.expected_action = expected_action # Semantic description

    def __repr__(self):
        return f"Problem(id={self.id}, log_path={self.log_path}, last_successful_action={self.last_successful_action})"

def get_problem(problem_id):
    """
    Given a problem id, return the problem object

    :param problem_id: The id of the problem
    :return: The problem object
    """
    event_log = pl.read_csv(problems[problem_id]["log_path"])
    robot_trace = event_log[problems[problem_id]["last_successful_action"]:]

    last_successful_action = event_log.row(problems[problem_id]["last_successful_action"] - 1, named=True)["EventDescription"]
    expected_action = event_log.row(problems[problem_id]["last_successful_action"], named=True)["EventDescription"]

    problem = Problem(
        id=problem_id,
        log_path=problems[problem_id]["log_path"],
        robot_trace=robot_trace,
        last_successful_action_idx=problems[problem_id]["last_successful_action"],
        last_successful_action=last_successful_action,
        robot_last_screenshot=problems[problem_id]["robot_last_screenshot"],
        expected_action=expected_action
    )
    
    return problem

# Custom system prompts for the RPA recovery scenario
RECOVERY_PLANNER_PROMPT = """
You are a specialized AI planner designed to recover robotic process automation (RPA) workflows that have failed.
Your role is to analyze the current state, understand what went wrong, and create a plan to get the process back on track.

You will be given:
1. The last successful action performed by the robot
2. The action that was expected to be performed but failed
3. The current screenshot of the application
4. Information about the overall process

Follow these guidelines:
1. Carefully analyze the last successful action and the expected action to understand where the process broke.
2. Examine the provided screenshot to assess the current UI state.
3. Determine what might have caused the failure (e.g., UI changes, timing issues, unexpected popups).
4. Create a precise step-by-step plan to recover the process and continue from where it left off.
5. The goal is to get the process back to a state where the robot can continue its automation.

Your response should be a JSON object with the following structure:
```json
{
  "reasoning": {
    "failure_analysis": "Analysis of what may have caused the failure",
    "ui_state": "Description of the current UI state and how it differs from expected",
    "recovery_approach": "General approach for recovery",
    "challenges": "Potential challenges or alternative approaches"
  },
  "steps": ["Step 1", "Step 2", "Step 3", "..."]
}
```

Your reasoning should include:
1. Analysis of what may have caused the failure
2. How the current UI state differs from what was expected
3. What steps are needed to recover and continue the process
4. Any potential challenges or alternative approaches

Remember that your plan will be executed by an action module capable of interacting with UI elements, so be specific about what elements to interact with and how.
"""

RECOVERY_ACTION_PROMPT = """
You are an AI designed to execute recovery actions for robotic process automation (RPA) workflows that have failed.
Your goal is to perform the specific actions needed to get the process back on track so the robot can continue its work.

You will be provided with:
1. A high-level recovery plan created by a planning AI
2. A history of past actions (if any have been taken during recovery)
3. A screenshot showing the current state of the application
4. Details about the last successful robot action and the action that was expected but failed

Your task is to determine the exact concrete action required to execute the current step in the recovery plan.
Focus on one atomic action based on the UI elements visible in the screenshot.

Guidelines:
1. Carefully examine the screenshot to identify UI elements relevant to the current step
2. Ground your action on observable elements in the UI
3. Provide clear execution details (clicks, keyboard input, etc.)
4. If an element isn't visible or the step cannot be completed, explain why and suggest alternatives

Your response should be a JSON object with the following structure:
```json
{
  "context_analysis": "Detailed explanation of your reasoning for identifying the action",
  "action": {
    "type": "LeftClick|RightClick|Type|Press|Finish|Scroll",
    "target": "Description of the target element or text to type"
  }
}
```

Possible action types:
- "LeftClick": Click on a UI element
- "RightClick": Right click on a UI element
- "Type": Type text into a field
- "Press": Press a specific key
- "Finish": Mark the task as complete
- "Scroll": Scroll in a specified direction (target should be "UP", "DOWN", "LEFT", or "RIGHT")

Remember that you are specifically trying to recover from a failure point in an RPA process, so focus on getting the workflow back to a state where the robot can continue its normal execution.
"""

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
        planner = QwenVLPlanner("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
        planner.manual_load()
        logger.debug("Initialized QwenVLPlanner for recovery")
        
        # Create prompt with problem-specific context
        prompt = f"""
        **Recovery Task**: Recover from a broken RPA process and get it back on track.
        
        **Last Successful Action**: {problem.last_successful_action}
        
        **Failed Action (Objective)**: {problem.expected_action}
        
        **Process Context**:
        This is part of a {problem.robot_trace[0]['ActivityLabel']} workflow.
        The robot was performing a sequence of UI interactions when it failed.
        
        **Goal**:
        Identify what went wrong and provide steps to recover the process so the robot can continue.
        """
        
        logger.debug(f"Generated recovery planning prompt for problem {problem.id}")
        
        # Get the recovery plan
        plan = planner.plan(
            RECOVERY_PLANNER_PROMPT,
            prompt,
            image=current_screenshot
        )

        # TODO: Use LLMs to validate the plan against a pre-established set of expected solutions for the problem
        
        logger.info(f"Generated recovery plan with {len(plan.steps)} steps")
        logger.debug(f"Recovery plan steps: {json.dumps(plan.steps, indent=2)}")
        
        planner.manual_unload()
        del planner
        torch.cuda.empty_cache()
        
        return plan
        
    except Exception as e:
        log_exception(logger, e, {
            "problem_id": problem.id,
            "last_successful_action": problem.last_successful_action,
            "expected_action": problem.expected_action
        })
        logger.error(f"Failed to plan recovery for problem {problem.id}")
        raise LoggedException()

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
        
        **Recovery Plan**: {', '.join(plan.steps)}.
        
        **Plan Reasoning**: {plan.reasoning}
        
        **History**:
        {history}
        
        **Result of Last Action**: {history.last_result}
        
        **Current Step to Execute**: {step}
        """
        
        # Generate the action
        action = action_model.action(
            RECOVERY_ACTION_PROMPT,
            prompt,
            image=current_screenshot
        )
        
        logger.info(f"Generated action: {action.action} with target: {action.action_target}")
        
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
        log_exception(logger, e, {
            "step": step,
            "problem_id": problem.id,
            "history_length": len(history.actions) if history else 0
        })
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

        image_path = os.path.join(os.path.dirname(problem.log_path), 
                                problem.robot_trace[0]['Screenshot'][0])
        
        logger.info(f"Loading screenshot from {image_path}")
        try:
            current_screenshot = Image.open(problem.robot_last_screenshot)
            logger.debug("Screenshot loaded successfully")
        except Exception as e:
            log_exception(logger, e, {"image_path": image_path})
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
                logger.info(f"Executing step {i+1} of {len(plan.steps)}: {step}")
                
                result = execute_recovery_step(
                    step,
                    history,
                    current_screenshot,
                    problem,
                    plan
                )
                
                if result != ActionResult.SUCCESS:
                    logger.warning(f"Step {i+1} failed with result {result}")
                    success = False
                    break
                
                # In a real implementation, we would update the screenshot here
                # For this toy problem, we'll simulate moving to the next screenshot
                next_idx = min(
                    problem.last_successful_action_idx + i + 1, 
                    len(problem.robot_trace) - 1
                )
                if next_idx < len(problem.robot_trace):
                    next_image_path = os.path.join(
                        os.path.dirname(problem.log_path),
                        problem.robot_trace[next_idx]['Screenshot'][0]
                    )
                    try:
                        current_screenshot = Image.open(next_image_path)
                        logger.debug(f"Updated screenshot to {next_image_path}")
                    except Exception as e:
                        log_exception(logger, e, {"image_path": next_image_path})
                        logger.warning(f"Could not load next screenshot {next_image_path}, continuing with current")
            
            if success:
                logger.info("Recovery successful!")
                return True
            
            # If we get here, recovery failed - try again with updated plan
            logger.warning(f"Recovery attempt {attempt + 1} failed, retrying")
            plan = plan_recovery(problem, current_screenshot)
            attempt += 1
        
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
    parser.add_argument("--problem_id", type=int, help="ID of the problem to solve", default=1)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging", default=False)
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum number of retry attempts"
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
