"""The "finance" skill: SKYE's Notion expense tracker."""

import re

from skills.base import Skill
from skills.finance.tools import account_balance, add_expense, budget_status, expense_summary, list_expenses
from skills.notion_client import undo_last_entry
from skills.routing_helpers import PERIOD, category_from, period_group

SKILL = Skill(
    name="finance",
    utterances=['how much money did I spend last month', 'what is my account balance', 'add an expense of twelve for lunch',
                'am I within budget', 'what did I buy this week', 'how much did I pay for rent'],
    pattern=re.compile(
        r"\b(spent|spend|spending|expenses?|budget|paid|bought|purchased|costs?|dollars?|bucks|"
        r"balance|income|salary|rent|groceries|savings|money|afford|transactions?)\b|\$\s?\d",
        re.IGNORECASE,
    ),
    manifest=(
        "Finance tools (his Notion finance tracker): "
        'expense_summary{"period","category"} · list_expenses{"period","category"} · '
        'add_expense{"name","amount","category"} · budget_status{} · account_balance{} · '
        "undo_last_entry{}. For any question about how much he spent (in total, on a category, "
        "or in a period) use expense_summary; use list_expenses only when he asks to see or "
        "list individual expenses. period can be \"this month\", \"last month\", a month name, "
        '"today", "this week" or "all time". Use add_expense when he says he spent or bought '
        "something, with amount as a plain number. Use undo_last_entry when he says undo, "
        "remove that or scratch that entry."
    ),
    examples=[
        ("how much have I spent this month?", '{"name": "expense_summary", "arguments": {"period": "this month"}}'),
        ("how much did I spend on food last month?", '{"name": "expense_summary", "arguments": {"period": "last month", "category": "food"}}'),
        ("I spent 12 dollars on lunch at Subway", '{"name": "add_expense", "arguments": {"name": "Subway lunch", "amount": "12", "category": "Food"}}'),
    ],
    tools=[expense_summary, list_expenses, add_expense, budget_status, account_balance, undo_last_entry],
    direct_routes=[
        ("expense_summary", re.compile(
            r"\b(?:how much (?:have i |did i |do i |i've )?(?:spent|spend)|(?:my )?(?:total )?(?:spending|expenses))\b(?:.*?\b" + PERIOD + r"\b)?", re.IGNORECASE),
         lambda m: {k: v for k, v in {"period": period_group(m) or "this month", "category": category_from(m.string)}.items() if v}),
    ],
)
