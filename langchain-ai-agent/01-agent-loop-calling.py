from dotenv import load_dotenv
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# =========================
# Tool Section
# =========================

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >>> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    if product not in prices:
        raise ValueError(f"Unknown product: {product}")

    return prices[product]


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the price and return the final price."""
    print(f"    >>> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}

    if discount_tier not in discount_percentages:
        raise ValueError(f"Invalid discount tier: {discount_tier}")

    discount = discount_percentages[discount_tier]
    return round(price * (1 - discount / 100), 2)


# =========================
# Agent Loop
# =========================

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Query: {question}")
    print("=" * 60)

    # -------------------------
    # Thought
    # -------------------------
    messages = [
        SystemMessage(
            content=(
                "You're a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price.\n"
                "   You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount after you have received a price from "
                "   get_product_price.\n"
                "3. Never calculate discounts yourself using math.\n"
                "   Always use the apply_discount tool.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # -------------------------
        # Answer
        # -------------------------
        if not tool_calls:
            print(f"\nAnswer: {ai_message.content}")
            return ai_message.content

        # -------------------------
        # Action
        # -------------------------
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Action: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools list")

        observation = tool_to_use.invoke(tool_args)

        # -------------------------
        # Observation
        # -------------------------
        print(f"Observation: {observation}")

        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id,
            )
        )

    raise RuntimeError("Agent reached MAX_ITERATIONS without producing a final answer.")


if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")