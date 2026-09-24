"""Finance tracker tools — reads/writes SKYE's Notion expense tracker. See
skills/notion_client.py for the shared plumbing this builds on."""

from datetime import date

from skills import canvas
from skills.base import SkillError
from skills.notion_client import (
    day, fuzzy_pick, join, match_name, money, need, paginate, api,
    parse_amount, parse_date, recent_undo_entries, undo_push, DUPLICATE_WINDOW_S,
)
from skills.notion_client import period as period_range

CATEGORY_ALIASES = {
    "groceries": "Food", "grocery": "Food", "lunch": "Food", "dinner": "Food", "breakfast": "Food",
    "coffee": "Food", "takeaway": "Food", "restaurant": "Food", "snack": "Food", "meal": "Food",
    "uber": "Transportation", "taxi": "Transportation", "train": "Transportation", "bus": "Transportation",
    "opal": "Transportation", "transport": "Transportation", "fuel": "Transportation", "petrol": "Transportation",
    "phone": "Utilities", "internet": "Utilities", "electricity": "Utilities", "water": "Utilities",
    "bills": "Utilities", "bill": "Utilities", "wifi": "Utilities", "laundry": "Utilities",
    "book": "Knowledge", "books": "Knowledge", "course": "Knowledge", "tuition": "Knowledge", "fees": "Knowledge",
    "netflix": "Entertainment", "movie": "Entertainment", "movies": "Entertainment", "game": "Entertainment",
    "games": "Entertainment", "spotify": "Entertainment", "concert": "Entertainment",
    "clothes": "Shopping", "clothing": "Shopping", "shoes": "Shopping",
    "stock": "Investing", "stocks": "Investing", "etf": "Investing", "shares": "Investing",
}
import time  # noqa: E402  (kept close to its one use, below)


def _categories():
    from skills.notion_client import title, cached
    def load():
        rows = paginate("POST", f"/databases/{need('categories')['id']}/query")
        return [{"id": r["id"], "name": title(r), "budget": r["properties"]["Monthly Budget"]["number"] or 0} for r in rows]
    return cached("categories", 300, load)


def _accounts():
    from skills.notion_client import title, cached
    return cached("accounts", 300, lambda: [
        {"id": r["id"], "name": title(r), "initial": r["properties"]["Initial Balance"]["number"] or 0}
        for r in paginate("POST", f"/databases/{need('accounts')['id']}/query")
    ])


def _expenses(start=None, end=None):
    from skills.notion_client import text
    conds = []
    if start:
        conds.append({"property": "Date", "date": {"on_or_after": start.isoformat()}})
    if end:
        conds.append({"property": "Date", "date": {"before": end.isoformat()}})
    body = {"filter": {"and": conds}} if len(conds) > 1 else ({"filter": conds[0]} if conds else {})
    cat_names = {c["id"]: c["name"] for c in _categories()}
    out = []
    for r in paginate("POST", f"/databases/{need('expenses')['id']}/query", body):
        p = r["properties"]
        rel = p["Category"]["relation"]
        out.append({
            "id": r["id"], "name": text(p["Expenses"]), "amount": p["Amount"]["number"] or 0,
            "date": (p["Date"]["date"] or {}).get("start", "")[:10],
            "category": cat_names.get(rel[0]["id"], "Uncategorised") if rel else "Uncategorised",
        })
    return out


def _scope(label):
    return {"all time": "Over all time", "today": "Today", "yesterday": "Yesterday",
            "this week": "This week", "last week": "Last week"}.get(label, f"In {label}")


def expense_summary(period="", category=""):
    """Total spent in a period (default this month) with a breakdown by category. period: "this month", "last month", "september", "today", "this week", "all time". category: optional, to total one category."""
    start, end, label = period_range(period)
    rows = _expenses(start, end)
    cat = None
    if category:
        names = [c["name"] for c in _categories()]
        cat = match_name(category, names, CATEGORY_ALIASES)
        if not cat:
            return f"I could not find a category called {category}. Your categories are {join(names)}."
        rows = [r for r in rows if r["category"] == cat]
    if not rows:
        return f"No expenses are recorded {'on ' + cat + ' ' if cat else ''}for {label}."
    total = sum(r["amount"] for r in rows)
    lead = (f"{_scope(label)}{' on ' + cat if cat else ''}, you have spent {money(total)} "
            f"across {len(rows)} expense{'s' if len(rows) != 1 else ''}.")
    if cat:
        return lead
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]
    ranked = sorted(by_cat.items(), key=lambda kv: -kv[1])
    text_out = f"{lead} By category: {join(f'{n} {money(v)}' for n, v in ranked[:5])}."
    return canvas.attach(text_out, f"Spending, {label}", canvas.bar_chart(f"By category, {label}", [n for n, _ in ranked[:8]], [round(v, 2) for _, v in ranked[:8]], "dollars"))


def list_expenses(period="", category="", limit="5"):
    """Lists the most recent individual expenses, optionally for a period or category."""
    start, end, label = period_range(period) if period else (None, None, "recently")
    rows = _expenses(start, end)
    if category:
        cat = match_name(category, [c["name"] for c in _categories()], CATEGORY_ALIASES)
        if cat:
            rows = [r for r in rows if r["category"] == cat]
    rows.sort(key=lambda r: r["date"], reverse=True)
    n = max(1, min(int(parse_amount(limit) or 5), 10))
    rows = rows[:n]
    if not rows:
        return "There are no matching expenses."
    parts = [f"{r['name']}, {money(r['amount'])}, {day(r['date'])}, {r['category']}" for r in rows]
    lead = "Your latest expense" if len(rows) == 1 else f"Your latest {len(rows)} expenses"
    return f"{lead}: " + "; ".join(parts) + "."


def budget_status() -> str:
    """This month's spending against each category's monthly budget."""
    from skills.notion_client import _month_start, _next_month
    start = _month_start(date.today())
    rows = _expenses(start, _next_month(start))
    spent = {}
    for r in rows:
        spent[r["category"]] = spent.get(r["category"], 0) + r["amount"]
    budgeted = [c for c in _categories() if c["budget"] > 0]
    total_spent = sum(spent.values())
    total_budget = sum(c["budget"] for c in budgeted)
    if not budgeted:
        return f"No monthly budgets are set. You have spent {money(total_spent)} so far this month."
    parts = []
    for c in budgeted:
        s = spent.get(c["name"], 0)
        state = "over budget" if s > c["budget"] else "within budget"
        parts.append(f"{c['name']} {money(s)} of {money(c['budget'])}, {state}")
    unbudgeted = sum(v for k, v in spent.items() if k not in {c["name"] for c in budgeted})
    tail = f" Spending in categories without a budget comes to {money(unbudgeted)}." if unbudgeted else ""
    return (f"This month you have spent {money(total_spent)} in total, against {money(total_budget)} budgeted. "
            + join(parts) + "." + tail)


def account_balance() -> str:
    """The current balance of the bank account in the finance tracker."""
    accounts = _accounts()
    if not accounts:
        return "I could not find an account."
    inc = sum(r["properties"]["Amount"]["number"] or 0
              for r in paginate("POST", f"/databases/{need('income')['id']}/query"))
    exp = sum(r["amount"] for r in _expenses())
    lines = []
    for a in accounts:
        bal = a["initial"] + inc - exp if len(accounts) == 1 else None
        lines.append(f"{a['name']} balance is {money(bal)}" if bal is not None else a["name"])
    return join(lines) + f", after {money(exp)} spent and {money(inc)} received in total."


def add_expense(name="", amount="", category="", date=""):
    """Logs an expense in the finance tracker. name: what it was for. amount: a number of dollars. category: e.g. Food, Rent, Transportation. date: optional, defaults to today."""
    name = (name or "").strip()
    value = parse_amount(amount)
    if not name or value is None or value <= 0:
        return "I need what it was for and how much it cost to log an expense."
    cats = _categories()
    cat = match_name(category, [c["name"] for c in cats], CATEGORY_ALIASES) or match_name(name, [c["name"] for c in cats], CATEGORY_ALIASES)
    if not cat:
        return f"Which category should that go under? Your categories are {join([c['name'] for c in cats])}."
    when = parse_date(date)
    accounts = _accounts()
    if not accounts:
        return "I could not find an account to log that against."

    # Guard against the same entry arriving twice (a re-sent or merged utterance).
    for e in recent_undo_entries():
        if (e["kind"] == "create" and e.get("sig") == [name.lower(), round(value, 2), when.isoformat()]
                and time.time() - e["at"] < DUPLICATE_WINDOW_S):
            return f"That looks like a duplicate of {e['label']}, so I have not added it again."

    props = {
        "Expenses": {"title": [{"text": {"content": name}}]},
        "Amount": {"number": round(value, 2)},
        "Date": {"date": {"start": when.isoformat()}},
        "Category": {"relation": [{"id": next(c["id"] for c in cats if c["name"] == cat)}]},
        "Account": {"relation": [{"id": accounts[0]["id"]}]},
        "Notes": {"rich_text": [{"text": {"content": "Logged by SKYE"}}]},
    }
    page = api("POST", "/pages", {"parent": {"database_id": need("expenses")["id"]}, "properties": props})
    label = f"{name}, {money(value)}"
    undo_push({"kind": "create", "page_id": page["id"], "label": label,
               "sig": [name.lower(), round(value, 2), when.isoformat()]})
    return f"Logged {name}, {money(value)}, under {cat} on {day(when.isoformat())}."
