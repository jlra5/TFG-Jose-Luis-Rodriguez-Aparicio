#-----------------------------------------------------------------------------------------------------------------------
# Block 1 - Imports
#-----------------------------------------------------------------------------------------------------------------------
import random
import time
import json
import gradio as gr
import logging
import warnings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool, InjectedToolCallId
from langgraph.graph.message import add_messages
from langgraph.types import Command
from operator import add
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, Optional, Any

LOG_NAME = "logfile.log"

logger = logging.getLogger(LOG_NAME)
logger.setLevel(logging.DEBUG)
logger.propagate = False

handler = logging.FileHandler(LOG_NAME)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

warnings.filterwarnings("ignore", category=UserWarning)
#-----------------------------------------------------------------------------------------------------------------------
# Block 2 - State Definition
#-----------------------------------------------------------------------------------------------------------------------

class State(TypedDict):
    '''
    Shared state amongst graph nodes.
    '''
    messages: Annotated[list, add_messages]
    instruction: Optional[str]
    order_id: Optional[int]
    product_id: Optional[int]
    customer_group: Optional[int]
    tool: Optional[str]
    missing_params: Optional[list]
    error_descr: Optional[str]
    total_tokens: Annotated[int, add]
    node_name: Optional[str]
    tool_definition: Optional[str]

#-----------------------------------------------------------------------------------------------------------------------
# Block 3 - Tool definition
#-----------------------------------------------------------------------------------------------------------------------
"""
Module to manage tools
"""
# Tool defintion (incl. business rule)
@tool
def delay_order(
    tool_call_id: Annotated[str, InjectedToolCallId],
    order_id: Optional[int] = Field(
        description=(
            "Order ID to update. "
            "Use ALWAYS the 'order_id' from INFO context when available. "
            "It is a 9‑digit number, e.g. 123456789. "
            "If not available in INFO or instruction, use null."
        )
    ),
    new_date: Optional[str] = Field(
        description=(
            "New delivery date for the order. "
            "Format: dd/mm/yyyy, e.g. 15/01/2026. "
            "Extract it from the INSTRUCTION text when present. "
            "If not explicitly provided, use null."
        )
    )
) -> Command:
    """
    Delay the delivery date of an existing order for flexible customers (Group 1 only).
    Use this tool only when the instruction implies a delay or future availability,
    and the customer_group in INFO is exactly '1'.
    """
    action = f"delay_order(order_id={order_id}, new_date={new_date})"
    number = random.random() #Simulating tool execution error
    if number > 0.8:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return Command(update={
                            "messages": [ToolMessage(content=f"Executing tool delay_order",
                                                    tool_call_id=tool_call_id)],
                            "error_descr": error_descr,
                            "tool_definition": action,
                            "node_name": "delay_order"
    })

# Business rule attachment
delay_order.__dict__["business_rule"] = (
    "delay_order: ONLY for Customer Group 1 (Flexible customers). "
    "Use this tool when the instruction implies a delay or future availability, "
    "such as 'available on dd/mm/yyyy', 'restock date', 'future availability', or 'delay'. "
    "If customer_group is '2' (Strict), NEVER use delay_order; a delay must be "
    "handled via immediate cancellation instead. "
    "Always read 'customer_group' from the INFO context; never assume it."
)

@tool
def cancel_order_line(
    tool_call_id: Annotated[str, InjectedToolCallId],
    order_id: Optional[int] = Field(
        description=(
            "Order ID whose line must be cancelled. "
            "Use ALWAYS the 'order_id' from INFO context when available. "
            "It is a 9‑digit number, e.g. 123456789. "
            "If not available in INFO or instruction, use null."
        )
    ),
    product_id: Optional[int] = Field(
        description=(
            "Product ID of the item to cancel from the order. "
            "Prefer the 'product_id' from INFO context. "
            "It is a 6‑digit number, e.g. 123456. "
            "If not available in INFO or instruction, use null."
        )
    )
) -> Command:
    """
    Cancel a specific product line from an order.
    This tool is mandatory for strict customers in Group 2 whenever a delay,
    future availability, stock issue, or explicit cancellation is mentioned.
    """
    action = f"cancel_order_line(order_id={order_id}, product_id={product_id})"
    number = random.random() #Simulating tool execution error
    if number > 0.8:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return Command(update={
                            "messages": [ToolMessage(content=f"Executing tool cancel_order_line",
                            tool_call_id=tool_call_id)],
                            "error_descr": error_descr,
                            "tool_definition": action,
                            "node_name": "cancel_order_line"
    })

# Business rule attachment
cancel_order_line.__dict__["business_rule"] = (
    "cancel_order_line: MANDATORY for Customer Group 2 (Strict / Non-Flexible customers) "
    "whenever the instruction mentions 'future availability', 'available on dd/mm/yy', "
    "'delay', 'out of stock', 'backorder', or similar stock issues. "
    "Group 2 customers NEVER accept delays—any stock or availability problem must be "
    "handled by cancelling the affected line using this tool. "
    "This tool can also be used for any customer group when the instruction explicitly "
    "requests cancellation (e.g. 'cancel this item', 'remove this line'). "
    "Do NOT assume the customer group; always read 'customer_group' from the INFO context."
)

@tool
def swap_products(
    tool_call_id: Annotated[str, InjectedToolCallId],
    order_id: Optional[int] = Field(
        description=(
            "Order ID where the product swap will occur. "
            "Use ALWAYS the 'order_id' from INFO context when available. "
            "It is a 9‑digit number, e.g. 123456789. "
            "If not available, use null."
        )
    ),
    old_product_id: Optional[int] = Field(
        description=(
            "Current product ID to be replaced. "
            "Prefer the 'product_id' from INFO context. "
            "It is a 6‑digit number, e.g. 123456. "
            "If not available in INFO or instruction, use null."
        )
    ),
    new_product_id: Optional[int] = Field(
        description=(
            "New product ID to use instead. "
            "Extract this 6‑digit numeric ID from the INSTRUCTION text "
            "(e.g. 'change to 654321'). "
            "If not provided, use null."
        )
    )
) -> Command:
    """
    Replace an existing product in an order with a new product.
    Use this tool whenever the instruction requests a product change,
    replacement, or use of a specific alternative product ID.
    """
    action = f"swap_products(order_id={order_id}, old_product_id={old_product_id}, new_product_id={new_product_id})"
    number = random.random() #Simulating tool execution error
    if number > 0.8:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return Command(update={
                            "messages": [ToolMessage(content=f"Executing tool swap_products",
                            tool_call_id=tool_call_id)],
                            "error_descr": error_descr,
                            "tool_definition": action,
                            "node_name": "swap_products"
    })

# Business rule attachment
swap_products.__dict__["business_rule"] = (
    "swap_products: Use when the instruction requests a product change or replacement, "
    "with wording such as 'change to product X', 'replace with X', "
    "'use product X instead', or 'use product 654321'. "
    "When a concrete alternative product ID is provided, this tool has PRIORITY over stock/delay "
    "tools (delay_order, cancel_order_line) and quantity tools (change_product_quantity)"
    "Product not available instruction is not valid for this tool."
    "The 'new_product_id' MUST be taken from the instruction text as a 6-digit ID. "
    "The 'old_product_id' and 'order_id' should be read from the INFO context whenever "
    "possible, using null only if they cannot be determined. "
    "Do NOT use this tool if the instruction only changes quantities or mentions "
    "credit/financial issues without specifying a replacement product."
)

@tool
def change_product_quantity(
    tool_call_id: Annotated[str, InjectedToolCallId],
    order_id: Optional[int] = Field(
        description=(
            "Order ID whose product quantity must be updated. "
            "Use ALWAYS the 'order_id' from INFO context when available. "
            "It is a 9‑digit number, e.g. 123456789. "
            "If not available, use null."
        )
    ),
    product_id: Optional[int] = Field(
        description=(
            "Product ID whose quantity will be changed. "
            "Prefer the 'product_id' from INFO context. "
            "It is a 6‑digit number, e.g. 123456. "
            "If not available in INFO or instruction, use null."
        )
    ),
    new_quantity: Optional[int] = Field(
        description=(
            "New quantity to set for this product. "
            "Extract the integer from the INSTRUCTION text when it mentions "
            "a specific quantity (e.g. 'reduce to 1000 units', 'available "
            "quantity 500', 'maximum qunatity 10 units). If not explicitly provided, use null."
        )
    )
) -> Command:
    """
    Update the ordered quantity of a product line.
    Use this tool when the instruction changes, reduces, limits,
    or sets a specific quantity, including 'maximum available quantity'.
    """
    action = f"change_product_quantity(order_id={order_id}, product_id={product_id}, new_quantity={new_quantity})"
    number = random.random() #Simulating tool execution error
    if number > 0.8:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return Command(update={
                            "messages": [ToolMessage(content=f"Executing tool change_product_quantity",
                            tool_call_id=tool_call_id)],
                            "error_descr": error_descr,
                            "tool_definition": action,
                            "node_name": "change_product_quantity"
    })

# Business rule attachment
change_product_quantity.__dict__["business_rule"] = (
    "change_product_quantity: Use when the instruction modifies, reduces, limits, "
    "or sets a specific quantity for a product, including phrases like "
    "'reduce quantity to X', 'set quantity to X', 'only X units left', "
    "'available quantity X', or 'maximum available quantity X'. "
    "Do NOT use this tool for pure stock/availability delays associated with a date "
    "(e.g. 'available on dd/mm/yy'); those should be handled with delay_order or "
    "cancel_order_line according to the customer_group. "
    "Do NOT use this tool for product replacements; use swap_products instead when "
    "the user specifies a different product ID. "
    "If order_id, product_id, or new_quantity cannot be derived from INFO or the "
    "instruction, they may be set to null, but the model should always try to fill "
    "as many parameters as possible from the available context."
)

@tool
def unblock_credit(
    tool_call_id: Annotated[str, InjectedToolCallId],
    order_id: Optional[int] = Field(
        description=(
            "Order ID whose credit block must be released. "
            "Use ALWAYS the 'order_id' from INFO context when available. "
            "It is a 9‑digit number, e.g. 123456789. "
            "If not available in INFO or instruction, use null."
        )
    )
) -> Command:
    """
    Release an order that is blocked due to credit limit issues.
    Use this tool when the instruction requests to liberate, release,
    or remove a credit block on the order.
    """
    action = f"unblock_credit(order_id={order_id})"
    number = random.random() #Simulating tool execution error
    if number > 0.8:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return Command(update={
                            "messages": [ToolMessage(content=f"Executing tool unblock_credit",
                            tool_call_id=tool_call_id)],
                            "error_descr": error_descr,
                            "tool_definition": action,
                            "node_name": "unblock_credit"
    })

# Business rule attachment
unblock_credit.__dict__["business_rule"] = (
    "unblock_credit: Use ONLY for orders blocked due to credit limit or credit hold. "
    "Trigger this tool when the instruction asks to free, release, or unblock an order "
    "because of credit, with phrases like 'liberate order', 'release the order', "
    "'remove credit block', 'unblock credit', 'release credit hold', or 'clean credit'. "
    "Customer group is irrelevant for this tool; it can be used for any customer_group "
    "as long as the problem is a credit block. "
    "Do NOT use this tool for stock, availability, quantity, or product replacement "
    "issues; use the corresponding tools (delay_order, cancel_order_line, "
    "swap_products, change_product_quantity) instead."
)

# Tools list
TOOLS = [
    delay_order,
    cancel_order_line,
    swap_products,
    change_product_quantity,
    unblock_credit
]

tools = {
    "delay_order": delay_order,
    "cancel_order_line": cancel_order_line,
    "swap_products": swap_products,
    "change_product_quantity": change_product_quantity
}

# Business rules list
def extract_business_rules(tools):
    return [
        tool.__dict__.get('business_rule', '')
        for tool in tools
        if 'business_rule' in tool.__dict__
    ]

logger.info("MAIN - Extracting business rules")
BUSINESS_RULES = "\n".join(extract_business_rules(TOOLS))

#-----------------------------------------------------------------------------------------------------------------------
# Block 4 - The model
#-----------------------------------------------------------------------------------------------------------------------

options = {
    'temperature': 0,
    'top_k': 1,
    'top_p': 0.1,
    'repeat_penalty': 1.2,
    'format': "json",
    'keep_alive': 0
}

logger.info("MAIN - Setting up model")
llm = ChatOllama(
    model="qwen2.5",
    options=options
)

llm_with_tools = llm.bind_tools(TOOLS, tool_choice="required")


# -----------------------------------------------------------------------------------------------------------------------
# Block 5 - Node definition
# -----------------------------------------------------------------------------------------------------------------------

# Model node
def model_node(state: State) -> dict:
    """
    Analyze customer input and extract tool & parameters.
    Assign None value in case no tool is found and/or no parameters are found.
    """
    # Gets the inputs: instruction, customer group, order id, product id
    # Use the model to infer the tool and its parameters analyzing tool_calls
    # Check tool_calls for unknown tools and missing parameters
    # Return: message, chosen tool, missing parameters
    logger.info(f"MODEL NODE - Starting")

    instruction = state["instruction"]
    customer_group = state["customer_group"]
    order_id = state["order_id"]
    product_id = state["product_id"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are a logistics assistant.

            ###DATA SOURCES:
            - INSTRUCTION: User's intent or request.
            - INFO: Authoritative context (Order ID, Product ID, Customer Group).

            ### CRITICAL BUSINESS RULES (YOU MUST OBEY STRICTLY):
            {BUSINESS_RULES}

            ###EXECUTION STEPS:
            1. Extract Customer Group from INFO.
            2. Match the instruction against the business rules above.
            3. Call the appropriate tool.

            ###CONSTRAINTS:
            1. Use only one tool each time. Do not concatenate them.
            2. If the tool is not clear, do not return any.

            INFO: Order {{order_id}}, Product {{product_id}}, Group {{customer_group}}
            INSTRUCTION: {{instruction}}
        """),
        ("human", "RESPOND ALWAYS with REASONING first, then tool call (if applicable).")
    ])

    try:
        logger.info("MODEL NODE - Invoking model")
        result = (prompt | llm_with_tools).invoke({"instruction": instruction,
                                                   "customer_group": customer_group,
                                                   "product_id": product_id,
                                                   "order_id": order_id})
        missing_params = []
        tool_name = ""
        tool_args = None

        if result.tool_calls:
            logger.info("MODEL NODE - Executing tool calls")
            tool_name = result.tool_calls[0]["name"]
            tool_args = result.tool_calls[0]["args"]
            missing_params = []
            for param, value in tool_args.items():
                if value is None:
                    missing_params.append(param)
        else:
            logger.info("MODEL NODE - No tool calls found")
            missing_params = []
            tool_name = None

        usage = result.response_metadata
        total_tokens = usage.get("eval_count", 0) + usage.get("prompt_eval_count", 0)

    except Exception as e:
        logger.info(f"MODEL NODE - Exception invoking model: {e}")
        print(f"Error: {e}")

    error_descr = ""
    if len(missing_params) > 0:
        error_descr = f"Missing parameters: {', '.join(missing_params)}"
    if tool_name is None:
        error_descr = error_descr.join("Unknown tool")
    logger.info(f"MODEL NODE - Quitting node - error: {error_descr}")

    return {
        "messages": [result],
        "tool": tool_name,
        "missing_params": missing_params,
        "error_descr": error_descr,
        "total_tokens": total_tokens,
        "node_name": "MODEL",
        "tool_definition": None
    }


# Tool Node
tool_node = ToolNode(tools=TOOLS)


# Human feedback
def human_feedback_node(state: State) -> dict:
    """
    Gives input to customer on missing or wrong data for tool or parameters
    """
    logger.info(f"HUMAN FEEDBACK NODE - Starting")
    logger.info(f"HUMAN FEEDBACK NODE - Feedback {state["error_descr"]}")
    logger.info(f"HUMAN FEEDBACK NODE - Quitting")
    return {
        "messages": f"Feedback given: {state['error_descr']} ",
        "node_name": "HUMAN FEEDBACK"
    }

# -----------------------------------------------------------------------------------------------------------------------
# Block 6 - Conditional routing
# -----------------------------------------------------------------------------------------------------------------------

def check_correct_tool(state: State) -> Literal["human_feedback", "tools"]:
    """
    Routes to ToolNode if a tool and all of its parameters are present.
    Routes to give feedback to humans in any other case.
    """
    logger.info(f"EDGE CHECK CORRECT TOOL - Starting")
    result = ""
    if len(state["error_descr"])>0:
        result = "human_feedback"
    else:
        result = "tools"
    logger.info(f"EDGE CHECK CORRECT TOOL - Routing to {result}")
    logger.info(f"EDGE CHECK CORRECT TOOL - Quitting")
    return result

def check_correct_execution(state: State) -> Literal["human_feedback", "end"]:
    """
    Routes to End if the tool is executed correctly.
    Routes to give feedback to humans in any other case.
    """
    logger.info(f"EDGE CHECK CORRECT EXECUTION- Starting")
    result = ""
    if len(state["error_descr"])>0:
        result = "human_feedback"
    else:
        result = "end"
    logger.info(f"EDGE CHECK CORRECT EXECUTION - Routing to {result}")
    logger.info(f"EDGE CHECK CORRECT EXECUTION - Quitting")
    return result
#-----------------------------------------------------------------------------------------------------------------------
# Block 7 - Building the graph
#-----------------------------------------------------------------------------------------------------------------------

logger.info(f"MAIN - Building Graph")

builder = StateGraph(State)

# Add nodes
builder.add_node("model", model_node)
builder.add_node("human_feedback", human_feedback_node)
builder.add_node("tools", tool_node)

# Define workflow
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model",
    check_correct_tool,
    {
        "human_feedback": "human_feedback",
        "tools": "tools"
    }
)
builder.add_conditional_edges(
    "tools",
    check_correct_execution,
    {
        "human_feedback": "human_feedback",
        "end": END
    }
)
builder.add_edge("tools", END)
builder.add_edge("human_feedback", END)

logger.info(f"MAIN - Compiling Graph")
graph = builder.compile()

#-----------------------------------------------------------------------------------------------------------------------
# Block 8 - Running examples
#-----------------------------------------------------------------------------------------------------------------------

def print_graph_structure(graph: StateGraph) -> None:
    """
    Prints on screen the graph structure.
    """
    print("-" * 70)
    print("GRAPH FLOW")
    print("-" * 70)
    print(graph.get_graph().draw_ascii())

def execute_instruction(graph:StateGraph, instruction: dict) -> None:
    """
    Graph execution for one single instruction.
    """
    initial_state = {
        "messages": HumanMessage(content="Execute instruction"),
        "instruction": instruction["instruction"],
        "order_id": instruction["order_id"],
        "product_id": instruction["product_id"],
        "customer_group": instruction["customer_group"],
        "tool": None,
        "missing_params": None,
        "total_tokens": 0
    }
    timer = time.perf_counter()
    result = graph.invoke(initial_state)
    timer = time.perf_counter() - timer
    hit = str(result["tool_definition"]) == str(instruction["correct_answer"])

    return {
        "hit": hit,
        "tokens": result['total_tokens'],
        "error_descr": result['error_descr'],
        "tool": result['tool'],
        "time": timer,
        "chosen_tool": result["tool_definition"]
    }

instructions = [
    {
        "order_id": 999999999,
        "product_id": 654321,
        "customer_group": 1,
        "instruction": "Check this, please!!!",
        "correct_answer": "None"
    },
    {
        "order_id": 197538624,
        "product_id": 234567,
        "customer_group": 1,
        "instruction": "Product available on 15/01/2026",
        "correct_answer": "delay_order(order_id=197538624, new_date=15/01/2026)"
    },
    {
        "order_id": 842673195,
        "product_id": 234567,
        "customer_group": 2,
        "instruction": "Product available on 15/01/2026",
        "correct_answer": "cancel_order_line(order_id=842673195, product_id=234567)"
    },
    {
        "order_id": 842673195,
        "product_id": 986532,
        "customer_group": 1,
        "instruction": "Product available on 15/01/2026, change to product 123456",
        "correct_answer": "swap_products(order_id=842673195, old_product_id=986532, new_product_id=123456)"
    },
    {
        "order_id": 842673195,
        "product_id": 784512,
        "customer_group": 2,
        "instruction": "Maximum quantity available 3000 units",
        "correct_answer": "change_product_quantity(order_id=842673195, product_id=784512, new_quantity=3000)"
    },
    {
        "order_id": 123456789,
        "product_id": 654321,
        "customer_group": 1,
        "instruction": "Product not available",
        "correct_answer": "None"
    },
    {
        "order_id": 123456789,
        "product_id": 654321,
        "customer_group":2,
        "instruction": "Product not available",
        "correct_answer": "cancel_order_line(order_id=123456789, product_id=654321)"
    },
    {
        "order_id": 987654321,
        "product_id": 123456,
        "customer_group": 1,
        "instruction": "Product not available till 20/01/2026",
        "correct_answer": "delay_order(order_id=987654321, new_date=20/01/2026)"
    },
    {
        "order_id": 987654321,
        "product_id": 123456,
        "customer_group": 1,
        "instruction": "Can't touch this!!!",
        "correct_answer": "None"
    },
    {
        "order_id": 987654321,
        "product_id": 123456,
        "customer_group": 1,
        "instruction": "Liberate order",
        "correct_answer": "unblock_credit(order_id=987654321)"
    }
]

def single_case_execution(graph: StateGraph)->None:
    """
    UI for one single instruction execution
    """

    def wrapper(order_id, product_id, customer_group, instruction, correct_answer):
        input_dict = {
            "order_id": order_id or "",
            "product_id": product_id or "",
            "customer_group": customer_group or "",
            "instruction": instruction or "",
            "correct_answer": correct_answer or ""
        }

        result = execute_instruction(graph, input_dict)

        with open(LOG_NAME, "r") as f:
            log = f.read()

        logger.info("="*70)    #to separate interactions

        return result["tokens"], f"{result["time"]:.2f}", log, result["chosen_tool"]

    with gr.Blocks(
        theme=gr.themes.Base(),
        css="""
            button { font-size: 16px !important; font-weight: bold !important; }
            .textbox { border: 2px solid #000 !important; }
            label { font-weight: bold !important; }
            .gradio-container footer { display: none !important; }
        """
    )  as TFGApp:
        gr.HTML("""
        <div style='text-align: center; background: #d4d4d8; color: #1f2937;
                    padding: 15px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='font-size: 20px; margin: 0;'>Single System Test</h3>
        </div>
        """)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Inputs")
                order_id = gr.Textbox(label="Order ID")
                product_id = gr.Textbox(label="Product ID")
                customer_group = gr.Textbox(label="Customer Group")
                instruction = gr.Textbox(label="Instruction")
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Outputs")
                tokens = gr.Textbox(label="Tokens")
                time = gr.Textbox(label="Time")
                chosen_tool = gr.Textbox(label="Tool")
        with gr.Row(equal_height=True):
            log = gr.Textbox(label="Log", lines=20)

        run_btn = gr.Button("Run")
        run_btn.click(wrapper,
                  inputs=[order_id, product_id, customer_group, instruction],
                  outputs=[tokens, time, log, chosen_tool]
        )
    TFGApp.launch(share=False)

def extended_test(graph: StateGraph, iterations: int, instructions: list) -> None:
    print("*" * 70)
    hits = 0
    tokens = 0
    time_model_invoking = 0
    timer = time.perf_counter()
    for i in range(iterations):
        counter = 0
        for instruction in instructions:
            counter += 1
            data = execute_instruction(graph, instruction)
            hits += data["hit"]
            tokens += data["tokens"]
            time_model_invoking += data["time"]
            print(f"Iteration: {i}.{counter} - Instruction: {instruction["instruction"]} Customer group: {instruction["customer_group"]} Chosen tool: {data["chosen_tool"]} - Correct answer: {instruction["correct_answer"]}")
        random.shuffle(instructions)
    timer = time.perf_counter() - timer

    print("*" * 70)
    print(f"RESULTS AFTER {iterations * len(instructions)} ITERATIONS:")
    total_instructions = iterations * len(instructions)
    print(f"Hits {hits} --> {hits/total_instructions:.1%}")
    print(f"Tokens {tokens} - Average: {tokens/total_instructions:.1f}")
    print(f"Total Model Time: {time_model_invoking}s - Average: {time_model_invoking/total_instructions:.1f}s")
    print(f"Total test time: {timer:.1f}s  Average: {timer/total_instructions:.1f}s ")


#print_graph_structure(graph)
#extended_test(graph, 2, instructions)
single_case_execution(graph)