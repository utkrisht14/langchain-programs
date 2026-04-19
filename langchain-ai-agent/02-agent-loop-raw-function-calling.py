from dotenv import load_dotenv

# ============================================================
# Environment Setup
# ============================================================
# Load variables from a .env file into the environment.

load_dotenv()

import ollama
from langsmith import traceable

# ============================================================
# Configuration
# ============================================================
# Maximum number of times the agent is allowed to loop.
# This prevents infinite loops if the model keeps calling tools.
MAX_ITERATIONS = 10

# Ollama model name to use for chat + tool calling.
MODEL = "qwen3:1.7b"


# ============================================================
# Tool Functions
# ============================================================
# These are plain Python functions.
# We trace them with LangSmith so each tool execution is visible.
#
# Unlike LangChain's @tool decorator, these remain normal functions.
# That means we must manually provide tool schemas to the LLM later.

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")

    # Fake in-memory product catalog
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    # Return 0 if the product is not found
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    # Discount lookup table
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    # Default to 0% if tier is unknown
    discount = discount_percentages.get(discount_tier, 0)

    # Apply discount and round to 2 decimals
    return round(price * (1 - discount / 100), 2)


# ============================================================
# Tool Schema for the LLM
# ============================================================
# Since we are NOT using LangChain's @tool decorator here,
# we must manually describe each function in JSON schema format.
#
# This schema tells the model:
# - the tool name
# - what the tool does
# - what arguments it accepts
# - which arguments are required
#
# This is basically the structure that @tool would generate for you.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE:
# Ollama can also derive tool schemas automatically if you pass the functions
# directly as tools, similar to LangChain's @tool decorator.
# Example:
#     tools_for_llm = [get_product_price, apply_discount]
#
# But for that to work well, your docstrings should follow Google-style format.
# We keep the manual JSON schema here because it makes the hidden tool metadata explicit.


# ============================================================
# Helper: Traced Ollama Call
# ============================================================
# This wraps ollama.chat() so the LLM call itself is visible in LangSmith.
#
# Input:
#   messages -> full conversation history
#
# Output:
#   response -> Ollama response object containing the assistant message
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ============================================================
# Agent Loop
# ============================================================
# This function implements the core loop of a tool-calling agent:
#
# 1. Build initial conversation messages
# 2. Send messages to the model
# 3. Check whether the model wants to call a tool
# 4. If yes, execute the tool
# 5. Append the result back into the conversation
# 6. Repeat until the model gives a final answer
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # --------------------------------------------------------
    # Tool Registry
    # --------------------------------------------------------
    # Map tool names (strings returned by the model)
    # to actual Python functions.
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # --------------------------------------------------------
    # Conversation / Prompt Messages
    # --------------------------------------------------------
    # These messages are the full context sent to the model.
    #
    # 1. System message -> instructions / rules for the assistant
    # 2. User message   -> the actual question from the user
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # --------------------------------------------------------
    # Main Agent Iteration Loop
    # --------------------------------------------------------
    # Each iteration does one of these:
    # - the model calls a tool
    # - or the model gives the final answer
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # ----------------------------------------------------
        # AI Message: Call the model with the full message history
        # ----------------------------------------------------
        # The model sees:
        # - the system prompt
        # - the user's question
        # - any previous assistant messages
        # - any previous tool results
        response = ollama_chat_traced(messages=messages)

        # Ollama returns a response object. The actual assistant message is here.
        ai_message = response.message

        # ----------------------------------------------------
        # Tool Calls Requested by the AI
        # ----------------------------------------------------
        # If the model wants to use a tool, those tool requests are stored here.
        tool_calls = ai_message.tool_calls

        # ----------------------------------------------------
        # Final Answer Case
        # ----------------------------------------------------
        # If there are no tool calls, then the model has answered directly.
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # ----------------------------------------------------
        # Tool Selection
        # ----------------------------------------------------
        # We intentionally process ONLY the first tool call in each iteration.
        # This keeps the flow simple: one tool call -> one result -> loop again.
        tool_call = tool_calls[0]

        # Ollama exposes tool call details via attribute access.
        # Example:
        #   tool_call.function.name
        #   tool_call.function.arguments
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # ----------------------------------------------------
        # Resolve Tool Name to Actual Python Function
        # ----------------------------------------------------
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # ----------------------------------------------------
        # Tool Execution / Observation
        # ----------------------------------------------------
        # Call the actual Python function using keyword arguments.
        # Example:
        #   apply_discount(price=1299.99, discount_tier="gold")
        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        # ----------------------------------------------------
        # Update Message History
        # ----------------------------------------------------
        # We append:
        # 1. The assistant's tool-call message
        # 2. The tool result message
        #
        # This lets the model continue from the updated state next iteration.
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    # --------------------------------------------------------
    # Safety Exit
    # --------------------------------------------------------
    # If the agent never produces a final answer, we stop here.
    print("ERROR: Max iterations reached without a final answer")
    return None


# ============================================================
# Script Entry Point
# ============================================================
# This block runs only when the file is executed directly.
# It will not run if the file is imported into another Python module.
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()

    # Example user question
    result = run_agent("What is the price of a laptop after applying a gold discount?")
