from dotenv import load_dotenv
from langchain.tools import tool
from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


# Loads environment variables from .env file.
# Example: OPENAI_API_KEY=...
load_dotenv()


# Safety limit: prevents the agent from running forever.
MAX_ITERATIONS = 10

# Model name used by LangChain.
# Format: provider:model-name
model = "openai:gpt-5"


# ============================================================
# TOOL SECTION
# ============================================================
# A tool is a normal Python function exposed to the LLM.
# The LLM cannot directly execute Python.
# It can only REQUEST a tool call by returning structured data.
# Your Python code then executes the tool.


@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.

    LangChain uses this docstring as the tool description.
    The LLM sees this description and decides when to call this tool.
    """

    print(f"    >>> Executing get_product_price(product='{product}')")

    # Fake product catalog.
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    if product not in prices:
        raise ValueError(f"Unknown product: {product}")

    return prices[product]


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price and return the final price.

    The LLM should call this only after it already has a price.
    """

    print(
        f"    >>> Executing apply_discount("
        f"price={price}, discount_tier='{discount_tier}')"
    )

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    if discount_tier not in discount_percentages:
        raise ValueError(f"Invalid discount tier: {discount_tier}")

    discount = discount_percentages[discount_tier]

    return round(price * (1 - discount / 100), 2)


# ============================================================
# AGENT LOOP
# ============================================================
# This is a manual tool-calling agent loop.
#
# Main pattern:
#
# User question
#   -> LLM decides next step
#   -> If tool needed, Python executes tool
#   -> Tool result is added back to messages
#   -> LLM sees result and continues
#   -> Final answer returned


@traceable(name="Shopping Agent Loop")
def run_agent(question: str):
    # --------------------------------------------------------
    # 1. Register available tools
    # --------------------------------------------------------
    # These are LangChain Tool objects because of @tool.
    # They are no longer just plain Python functions.
    tools = [get_product_price, apply_discount]

    # Each LangChain tool has a .name attribute.
    #
    # Example:
    # get_product_price.name == "get_product_price"
    # apply_discount.name == "apply_discount"
    #
    # This dictionary maps:
    #
    # {
    #   "get_product_price": <Tool object for get_product_price>,
    #   "apply_discount": <Tool object for apply_discount>
    # }
    #
    # Why needed?
    # The LLM returns only the tool name as text.
    # We use this dictionary to find the actual executable tool.
    tool_dict = {t.name: t for t in tools}

    # --------------------------------------------------------
    # 2. Create the chat model
    # --------------------------------------------------------
    # temperature=0 makes behavior more deterministic.
    llm = init_chat_model(model, temperature=0)

    # --------------------------------------------------------
    # 3. Bind tools to the LLM
    # --------------------------------------------------------
    # bind_tools tells the model:
    # "These tools exist, with these names, descriptions, and arguments."
    #
    # LangChain converts tools into JSON-schema-like descriptions.
    #
    # Conceptually, the model sees something like:
    #
    # tools = [
    #   {
    #     "name": "get_product_price",
    #     "description": "Look up the price of a product in the catalog.",
    #     "parameters": {
    #       "product": "string"
    #     }
    #   },
    #   {
    #     "name": "apply_discount",
    #     "description": "Apply a discount to the price...",
    #     "parameters": {
    #       "price": "number",
    #       "discount_tier": "string"
    #     }
    #   }
    # ]
    llm_with_tools = llm.bind_tools(tools)

    print(f"Query: {question}")
    print("=" * 60)

    # --------------------------------------------------------
    # 4. Prepare initial messages
    # --------------------------------------------------------
    # Messages are the conversation history.
    #
    # SystemMessage = instructions/rules for the model.
    # HumanMessage  = user's question.
    # AIMessage     = model's response, created later.
    # ToolMessage   = tool result, created after executing a tool.
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

    # --------------------------------------------------------
    # 5. Agent loop
    # --------------------------------------------------------
    # Use range(...), not:
    #
    # for iteration in (1, MAX_ITERATIONS + 1):
    #
    # because that only loops over two values: 1 and 11.
    #
    # Correct:
    # range(1, MAX_ITERATIONS + 1)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # ----------------------------------------------------
        # 6. Ask the LLM what to do next
        # ----------------------------------------------------
        # This is the main decision-making line.
        #
        # Input:
        # messages = [
        #   SystemMessage(...),
        #   HumanMessage(...),
        #   AIMessage(...),      # maybe from previous iteration
        #   ToolMessage(...),    # maybe from previous iteration
        # ]
        #
        # Output:
        # ai_message = AIMessage(...)
        #
        # AIMessage may contain either:
        #
        # Case A: final answer
        # ai_message.content = "The final price is $1000.99"
        # ai_message.tool_calls = []
        #
        # Case B: tool request
        # ai_message.content = ""
        # ai_message.tool_calls = [
        #   {
        #     "name": "get_product_price",
        #     "args": {"product": "laptop"},
        #     "id": "call_1"
        #   }
        # ]
        ai_message = llm_with_tools.invoke(messages)

        # Add the model response to conversation history.
        #
        # This is important because the next LLM call should know
        # what it previously requested.
        messages.append(ai_message)

        # Extract tool calls from AIMessage.
        #
        # tool_calls is a list because the model may request
        # one or more tool calls.
        tool_calls = ai_message.tool_calls

        # ----------------------------------------------------
        # 7. If no tool call, this is the final answer
        # ----------------------------------------------------
        # If tool_calls is empty, the model is done.
        #
        # Example:
        # ai_message.content = "The laptop costs $1000.99 after gold discount."
        # ai_message.tool_calls = []
        if not tool_calls:
            print(f"\nAnswer: {ai_message.content}")
            return ai_message.content

        # ----------------------------------------------------
        # 8. Pick the tool call to execute
        # ----------------------------------------------------
        # Because tool_calls returns a list, we pick the first item.
        #
        # Example shape:
        #
        # ai_message.tool_calls = [
        #   {
        #     "name": "get_product_price",
        #     "args": {"product": "laptop"},
        #     "id": "call_1"
        #   }
        # ]
        #
        # tool_calls[0] gives:
        #
        # {
        #   "name": "get_product_price",
        #   "args": {"product": "laptop"},
        #   "id": "call_1"
        # }
        #
        # Note:
        # This example handles only the first tool call.
        # A production version may loop through all tool_calls.
        tool_call = tool_calls[0]

        # Extract dictionary values from the selected tool call.
        #
        # tool_name:
        #   The name of the tool the LLM wants to call.
        #
        # tool_args:
        #   The arguments the LLM wants to pass to the tool.
        #
        # tool_call_id:
        #   Unique ID used to connect the tool result back
        #   to the exact tool request.
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"Action: {tool_name} with args {tool_args}")

        # ----------------------------------------------------
        # 9. Convert tool name into actual executable tool
        # ----------------------------------------------------
        # The LLM only gave us a string:
        #
        # tool_name = "get_product_price"
        #
        # But Python needs the actual Tool object.
        #
        # tool_dict looks like:
        #
        # {
        #   "get_product_price": <Tool object>,
        #   "apply_discount": <Tool object>
        # }
        #
        # So this line means:
        #
        # "Find the executable tool object whose name matches
        #  the name requested by the model."
        if tool_name not in tool_dict:
            raise ValueError(f"Tool '{tool_name}' not found in tool_dict")

        tool_to_use = tool_dict[tool_name]

        # ----------------------------------------------------
        # 10. Execute the selected tool
        # ----------------------------------------------------
        # tool_args is a dictionary.
        #
        # Example:
        # tool_args = {"product": "laptop"}
        #
        # invoke(tool_args) calls:
        #
        # get_product_price(product="laptop")
        #
        # The result is called an observation in agent terminology.
        observation = tool_to_use.invoke(tool_args)

        # ----------------------------------------------------
        # 11. Observation
        # ----------------------------------------------------
        # observation is the real output from the Python tool.
        #
        # Example:
        # observation = 1299.99
        print(f"Observation: {observation}")

        # ----------------------------------------------------
        # 12. Send tool result back to the LLM
        # ----------------------------------------------------
        # The LLM cannot automatically see Python return values.
        # We must add the tool result back into messages.
        #
        # ToolMessage shape conceptually:
        #
        # {
        #   "role": "tool",
        #   "content": "1299.99",
        #   "tool_call_id": "call_1"
        # }
        #
        # tool_call_id is important because it tells the model:
        # "This result belongs to the tool call you made earlier."
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id,
            )
        )

    # --------------------------------------------------------
    # 13. Safety fallback
    # --------------------------------------------------------
    # If the loop finishes without a final answer,
    # something went wrong or the model kept asking for tools.
    raise RuntimeError(
        "Agent reached MAX_ITERATIONS without producing a final answer."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
# This block runs only when this file is executed directly.
# It will not run if this file is imported into another file.
if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")