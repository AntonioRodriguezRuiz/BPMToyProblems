"""
Prompts module for ToyProblems.

This module contains prompts used for planner and action models.
"""

from .action_prompts import SYS_PROMPT_MID
from .planner_prompts import SYS_PROMPT_COT

__all__ = [
    'SYS_PROMPT_MID',
    'SYS_PROMPT_COT'
]