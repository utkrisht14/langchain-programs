"""
MANUAL REACT-STYLE AGENT WITH OLLAMA
====================================

What this program demonstrates
------------------------------
This is a small "agent" program.

An agent is just a loop that does this:

1. Ask the LLM what to do next
2. Read the LLM's answer
3. If the LLM wants to use a tool, run that tool in Python
4. Give the tool result back to the LLM
5. Repeat until the LLM gives a final answer

Important idea:
---------------
This version does NOT use native structured tool-calling.

Instead:
- we describe tools inside the prompt as plain text
- the LLM writes tool requests in a fixed text format
- Python parses that text using regex
- Python executes the tool manually

So this is a "manual agent loop".
"""

# ============================================================
# 1. IMPORTS
# ============================================================

# re      -> used to parse "Action:" and "Action Input:" from the LLM's text
# inspect -> used to read function signatures and docstrings automatically
import re
import inspect

# load_dotenv -> loads environment variables from a .env file
from dotenv import load_dotenv

# Load API keys / environment settings before using Ollama / tracing
load_dotenv()

import ollama
from langsmith import traceable


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Safety limit so the agent does not loop forever
MAX_ITERATIONS = 10

# The Ollama model we want to use
MODEL = "qwen3:1.7b"


# ============================================================
# 3. TOOLS
# ============================================================
#
# Tools are just normal Python functions.
# The LLM does NOT directly execute Python.
# Instead, the LLM writes which tool it wants to use,
# and then OUR code calls the Python function.

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f"    >> Executing get_product_price(product='{product}')")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    # Return 0 if product is not found
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.

    Available tiers:
    - bronze
    - silver
    - gold
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    # LLM text parsing usually gives us strings, so convert price to float
    price = float(price)

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    # If discount tier is unknown, use 0%
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# Dictionary: tool name -> actual Python function
# This lets us turn text like "apply_discount" into a real function call
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ============================================================
# 4. BUILD TOOL DESCRIPTIONS FOR THE PROMPT
# ============================================================
#
# Since we are NOT using structured JSON tool definitions here,
# we must explain the tools to the LLM using plain text.
#
# To avoid writing tool docs manually, we inspect the Python
# functions and generate descriptions automatically.

def get_tool_descriptions(tools_dict: dict) -> str:
    """
    Build a readable tool description string from the actual Python functions.

    Example output:
        get_product_price(product: str) -> float - Look up the price...
        apply_discount(price: float, discount_tier: str) -> float - Apply a discount...
    """
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        # Some decorators wrap the original function.
        # __wrapped__ gives access to the original function if available.
        original_function = getattr(tool_function, "__wrapped__", tool_function)

        # Read function parameters and return type
        signature = inspect.signature(original_function)

        # Read the function's docstring
        docstring = inspect.getdoc(tool_function) or ""

        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())


# ============================================================
# 5. THE PROMPT TEMPLATE
# ============================================================
#
# This is the most important idea in this version:
# the LLM is taught to behave like an agent ONLY through prompt instructions.
#
# We tell it:
# - what rules it must follow
# - what tools exist
# - what output format it must produce
#
# This style is often called "ReAct":
# Thought -> Action -> Observation -> repeat

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following question as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format exactly:

Question: the input question you must answer
Thought: think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# ============================================================
# 6. WRAPPER AROUND OLLAMA CHAT
# ============================================================
#
# We keep this small helper so the LLM call itself is traced in LangSmith.
#
# Notice:
# We are NOT passing tools=...
# This means Ollama does not know about tools in any structured way.
# It only sees text instructions in the prompt.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model: str, messages: list, options: dict):
    """
    Make one LLM call to Ollama.
    """
    return ollama.chat(
        model=model,
        messages=messages,
        options=options,
    )


# ============================================================
# 7. MAIN AGENT LOOP
# ============================================================
#
# This function coordinates everything.
#
# Sequence:
# 1. Build the initial prompt
# 2. Ask the model what to do
# 3. If final answer exists -> return it
# 4. Otherwise parse Action / Action Input
# 5. Execute the tool
# 6. Append Observation to scratchpad
# 7. Repeat

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    """
    Run the manual agent loop for a user question.
    """
    print(f"Question: {question}")
    print("=" * 60)

    # --------------------------------------------------------
    # A. INITIAL PROMPT
    # --------------------------------------------------------
    #
    # Instead of using separate system/user messages,
    # this version uses one large prompt string.
    prompt = react_prompt.format(question=question)

    # scratchpad = growing text memory
    #
    # This stores prior Thought / Action / Observation steps.
    # Every loop we resend the full prompt + scratchpad.
    scratchpad = ""

    # --------------------------------------------------------
    # B. LOOP
    # --------------------------------------------------------
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Combine the original prompt and all previous steps
        full_prompt = prompt + scratchpad

        # ----------------------------------------------------
        # C. ASK THE LLM
        # ----------------------------------------------------
        #
        # We pass the entire current state as one user message.
        #
        # stop=["\nObservation"] is very important:
        # It prevents the model from inventing a fake Observation.
        # The real Observation must come from our Python tool call.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            options={
                "stop": ["\nObservation"],
                "temperature": 0,
            },
        )

        # The text generated by the model
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # ----------------------------------------------------
        # D. CHECK FOR FINAL ANSWER
        # ----------------------------------------------------
        #
        # If the LLM already produced:
        #   Final Answer: ...
        # then we are done.
        print("  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()

            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")

            return final_answer

        # ----------------------------------------------------
        # E. PARSE TOOL REQUEST
        # ----------------------------------------------------
        #
        # If there is no final answer yet, we expect the model
        # to output something like:
        #
        # Action: get_product_price
        # Action Input: laptop
        #
        # We parse both lines using regex.
        print("  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with raw args: {tool_input_raw}")

        # ----------------------------------------------------
        # F. CONVERT TEXT INPUT INTO PYTHON ARGUMENTS
        # ----------------------------------------------------
        #
        # The prompt says Action Input should be comma-separated.
        #
        # Examples:
        #   laptop
        #   1299.99, gold
        #   price=1299.99, discount_tier=gold
        #
        # This logic accepts both plain values and key=value style.
        raw_args = [item.strip() for item in tool_input_raw.split(",")]
        args = [item.split("=", 1)[-1].strip().strip("'\"") for item in raw_args]

        # ----------------------------------------------------
        # G. EXECUTE THE TOOL
        # ----------------------------------------------------
        print(f"  [Tool Executing] {tool_name}{tuple(args)}")

        if tool_name not in tools:
            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools.keys())}"
            )
        else:
            # This calls the real Python function
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # ----------------------------------------------------
        # H. UPDATE SCRATCHPAD (AGENT MEMORY)
        # ----------------------------------------------------
        #
        # We append the model's output and the real observation.
        #
        # Example:
        #   Thought: I need the product price first
        #   Action: get_product_price
        #   Action Input: laptop
        #   Observation: 1299.99
        #   Thought:
        #
        # Next iteration the model sees all prior work.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    # --------------------------------------------------------
    # I. LOOP FAILED / STOPPED
    # --------------------------------------------------------
    print("ERROR: Max iterations reached without a final answer")
    return None


# ============================================================
# 8. SCRIPT ENTRY POINT
# ============================================================
#
# This block runs only when this file is executed directly.
# It will not run if the file is imported from another file.

if __name__ == "__main__":
    print("Hello Manual ReAct Agent with Ollama!")
    print()

    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )

    print(f"\nReturned result: {result}")