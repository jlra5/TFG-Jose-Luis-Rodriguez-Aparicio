from __future__ import annotations
#-----------------------------------------------------------------------------------------------------------------------
# Block 0 - Imports
#-----------------------------------------------------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langgraph.graph.message import add_messages
from operator import add
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal, Optional, Any

import json
import time
import gradio as gr
import random
import logging
import warnings
import joblib
import pandas as pd
import numpy as np

ERROR_SIM = 1 # To simulate wrong tool execution

MODEL_PATH = "intent_classifier.joblib"
THRESHOLDS = {
    "delay_order": 0.55,
    "cancel_order_line": 0.40,
    "swap_products": 0.45,
    "change_product_quantity": 0.45,
    "unblock_credit": 0.3,
    "Unknown": 0.0
}
AMBIGUITY_DIFF = 0.0

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
    messages: Annotated[list, add_messages]     # System messages

    instruction: Optional[str]
    order_id: Optional[int]
    product_id: Optional[int]
    customer_group: Optional[int]

    intention: Optional[str]                    # Identified instruction
    chosen_tool: Optional[str]                  # Matching tool to instruction
    tool_confidence: Optional[float]            # % Confidence of the tool
    tool_params: Optional[dict]                 # Parameters of the tool
    params_schema: Optional[str]
    total_tokens: Annotated[int, add]
    error_descr: Optional[str]
    tool_definition: Optional[str]


#-----------------------------------------------------------------------------------------------------------------------
# Block 3 - Tool definition
#-----------------------------------------------------------------------------------------------------------------------

# Tool Schemas for each tool

class ParamsDelayOrder(BaseModel):
    reasoning: str = Field(description="Briefly explain the extracted parameters"),
    order_id: int = Field(description="id of the order to be updated")
    new_date: str = Field(description="new updated delivery date")

class ParamsCancelOrderLine(BaseModel):
    reasoning: str = Field(description="Briefly explain the extracted parameters"),
    order_id: int = Field(description="id of the order to be updated")
    product_id: int = Field(description="id of the product whose quantity is to be set to zero")

class ParamsSwapProducts(BaseModel):
    reasoning: str = Field(description="Briefly explain the extracted parameters"),
    order_id: int = Field(description="id of the order to be updated")
    old_product_id: int = Field(description="id of the old product to be replaced")
    new_product_id: int = Field(description="id of the new product to be inserted")

class ParamsChangeProductQuantity(BaseModel):
    reasoning: str = Field(description="Briefly explain the extracted parameters"),
    order_id: int = Field(description="id of the order to be updated")
    product_id: int = Field(description="id of the product whose quantity is to be changed")
    new_quantity: int = Field(description="new quantity to be changed")

class ParamsUnblockCredit(BaseModel):
    reasoning: str = Field(description="Briefly explain the extracted parameters"),
    order_id: int = Field(description="id of the order to be updated")

class ParamsChangeUnknown(BaseModel):
    pass

TOOL_SCHEMAS = {
    "delay_order": ParamsDelayOrder,
    "cancel_order_line": ParamsCancelOrderLine,
    "swap_products": ParamsSwapProducts,
    "change_product_quantity": ParamsChangeProductQuantity,
    "unblock_credit": ParamsUnblockCredit,
    "unknown": ParamsChangeUnknown
}

# Define tools

@tool
def delay_order(
        order_id: int = Field(description="id of the order to be updated. It is a 9 digit number. Example:123456789"),
        new_date: str= Field(description="new updated delivery date. It is coded as dd/mm/yy. Example: 15/01/26")
) -> dict:
    """
    Changes the delivery date of the order with order_id to the date new_date.
    Use it when the intention is to delay the order.
    """
    action = f"delay_order(order_id={order_id}, new_date={new_date})"
    number = random.random() #Simulating tool execution error
    if number > ERROR_SIM:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return {
        "error_descr": error_descr,
        "chosen_tool": action
    }

@tool
def cancel_order_line(
        order_id: int = Field(description="id of the order to be updated. It is a 9 digit number. Example:123456789"),
        product_id: int = Field(description="""id of the product whose quantity is to be set to zero. Is is a 6 digit
                                            number. Example:123456""")
) -> str:
    """
    Cancels the order line belonging to the product. This means to set the product quantity to zero.
    Use it when the intention is to cancel the order.
    """
    action = f"cancel_order_line(order_id={order_id}, product_id={product_id})"
    number = random.random() #Simulating tool execution error
    if number > ERROR_SIM:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return {
        "error_descr": error_descr,
        "chosen_tool": action
    }

@tool
def swap_products(
        order_id: int = Field(description="id of the order to be updated. It is a 9 digit number. Example:123456789"),
        old_product_id: int = Field(description="id of the old product to be replaced. It is a 6 digit number. Example:123456") ,
        new_product_id: int = Field(description="id of the new product to be inserted. It is a 6 digit number. Example:123456")
) -> str:
    """
    Swaps old product with the new product in the order.
    Use it when the intention is to swap products in an order
    """
    action = f"swap_products(order_id={order_id}, old_product_id={old_product_id}, new_product_id={new_product_id})"
    number = random.random() #Simulating tool execution error
    if number > ERROR_SIM:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return {
        "error_descr": error_descr,
        "chosen_tool": action
    }


@tool
def change_product_quantity(
        order_id: int = Field(description="id of the order to be updated. It is a 9 digit number. Example:123456789"),
        product_id: int = Field(description="id of the product whose quantity is to be changed. It is a 6 digit number. Example:123456"),
        new_quantity: int = Field(description="Quantity to be inserted. It is a number. Example: 3000 units")
)-> str:
    """
    Updates the product quantity of the order line belonging to the product.
    Use it when the intention is to change the product quantity.
    """
    action = f"change_product_quantity(order_id={order_id}, product_id={product_id}, new_quantity={new_quantity})"
    number = random.random() #Simulating tool execution error
    if number > ERROR_SIM:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - Error: {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return {
        "error_descr": error_descr,
        "chosen_tool": action
    }

@tool
def unblock_credit(
    order_id: Optional[int] = Field(
        description=("Order ID whose credit block must be released. ")
    )
) -> str:
    """
    Release an order that is blocked due to credit limit issues.
    Use this tool when the instruction requests to liberate, release,
    or remove a credit block on the order.
    """
    action = f"unblock_credit(order_id={order_id})"
    number = random.random() #Simulating tool execution error
    if number > ERROR_SIM:
        error_descr = f"Error - {action}"
        logger.info(f"TOOL EXECUTION - {error_descr}")
    else:
        error_descr = ""
        logger.info(f"TOOL EXECUTION - Executing {action}")

    return {
        "error_descr": error_descr,
        "chosen_tool": action
    }

tools = {
    "delay_order": delay_order,
    "cancel_order_line": cancel_order_line,
    "swap_products": swap_products,
    "change_product_quantity": change_product_quantity,
    "unblock_credit": unblock_credit
}


#-----------------------------------------------------------------------------------------------------------------------
# Block 3 - Instantiate the model and bind tools
#-----------------------------------------------------------------------------------------------------------------------

options = {
    'temperature': 0,
    'top_k': 1,
    'top_p': 0.1,
    'repeat_penalty': 1.2
}

logger.info("MAIN - Setting up model")
llm = ChatOllama(
    #model="mistral",
    model="qwen2.5:1.5b",    #This model is faster but halucinates parameters when forced to choose
    #model="qwen2.5",
    options=options
)

model = joblib.load(MODEL_PATH)
classes = model.classes_
logger.info("MAIN - Logistic Regression Model loeaded")

# Función de inferencia con umbrales (retorna diccionario)
def predict_intent(df: pd.DataFrame) -> dict:
    """
    df debe tener columnas:
    - 'instruction'
    - 'customer_group'

    Retorna:
    {
        idx0: {'instruction': ..., 'customer_group': ..., 'predicted_intent': ..., 'confidence': ...},
        idx1: {...},
        ...
    }
    """
    predictions_raw = model.predict(df)
    probabilities = model.predict_proba(df)

    results_dict = {}

    for i, row in df.iterrows():
        probs = probabilities[i]
        sorted_idx = np.argsort(probs)[::-1]  # de mayor a menor
        top_idx = sorted_idx[0]
        second_idx = sorted_idx[1]

        top_class = classes[top_idx]
        top_prob = probs[top_idx]
        second_prob = probs[second_idx]

        # Aplicar umbral de clase
        class_threshold = THRESHOLDS.get(top_class, 0.6)
        # Decisión con umbrales y ambigüedad
        if top_class == "Unknown":
            final_class = "Unknown"
        elif top_prob < class_threshold:
            final_class = "Unknown"
        elif (top_prob - second_prob) < AMBIGUITY_DIFF:
            final_class = "Unknown"
        else:
            final_class = top_class

        logger.info(f"PREDICT INTENT - Top class: {top_class} Top class: {top_prob}  Second class: {classes[second_idx]} Confidence second class: {second_prob} Threshold: {class_threshold}")

        results_dict[i] = {
            "instruction": row["instruction"],
            "customer_group": row["customer_group"],
            "predicted_intent": final_class,
            "confidence": float(top_prob),
            "correct_answer": row["correct_answer"]
        }

    return results_dict

# Schema to force the output of the call to the SLM/LLM
# Reasoning to improve order bias
class Intention(BaseModel):
    """ Schema to restrict/focus SLM/LLM answers"""
    reasoning: str = Field(description="Briefly explain the extracted customer group and availability status derived from the instruction."),
    intention: Literal["delay_order", "cancel_order_line", "swap_products", "change_product_quantity", "unblock_credit", "unknown"]
    tool_confidence: float

#-----------------------------------------------------------------------------------------------------------------------
# Block 5 - Node definition
#-----------------------------------------------------------------------------------------------------------------------

def node_read_instruction(state: State) -> dict:
    """
    Node to read the instructions and set their intention.
    """

    # Reads the instruction field of the message
    # Uses the model (SLM/Embeddings/other) to find the most suitable tool and its confidence
    # structured_output could be used to limit model's variability
    # If no tool is found, return "Unknown"

    logger.info(f"MODEL NODE - Starting")
    error_descr = ""
    messages = state["messages"]
    if not messages:
        looger.info(f"MODEL NODE - No messages")
        return {
            "messages": [AIMessage(content=f"Intention: Unknown")],
            "instruction": "Unknown",
            "error_descr": error_descr,}

    instruction = state["instruction"]
    customer_group = state["customer_group"]

    new_data = pd.DataFrame([
        {
            "instruction": instruction,
            "customer_group": int(customer_group),
            "correct_answer": ""
        }
    ])
    logger.info("MODEL NODE - Invoking model")
    #intention = model.predict(new_data)[0]
    #tool_confidence = model.predict_proba(new_data).max()
    prediction = predict_intent(new_data)
    intention = prediction[0]["predicted_intent"]
    tool_confidence = prediction[0]["confidence"]
    logger.info(f"MODEL NODE - Model results: intention {intention} confidence {tool_confidence}")
    if intention.lower() == "unknown":
        error_descr = "Intention: Unknown"

    logger.info(f"MODEL NODE - Quitting node")

    return {
        "instruction": instruction,
        "intention": intention,
        "tool_confidence": tool_confidence,
        "messages": [AIMessage(content=f"Intention: {intention}")],
        "total_tokens": 0,
        "error_descr": error_descr,
        "node_name": "MODEL"
    }

def node_choose_tool(state: State) -> dict:
    """
    Node to choose the tool to apply to each instruction.
    """

    # Gets intention from previous node. If no intention identified, return "no tool"
    # Gets the schema of the identified tool
    # Uses the schema to build the prompt sent to the slm
    # if the intention->tool selection is clear, use the tool's schema to request a structured_output
    # Returns the parameters according to the schema

    logger.info(f"CHOOSE TOOL NODE - Starting")

    intention = state["intention"]

    error_descr = ""

    if intention.lower() == "unknown":
        logger.info(f"CHOOSE TOOL NODE - Intention unknown")
        error_descr = "Unknown intention"
        return {
            "tool_params": None,
            "error_descr": error_descr,
            "messages": AIMessage(content = "Unknown intention"),
            "node_name": "CHOOSE TOOL"
        }

    instruction = state["instruction"]
    tool_confidence = state["tool_confidence"]
    order_id = state["order_id"]
    product_id = state["product_id"]

    schema = TOOL_SCHEMAS[intention]
    schema_dict = schema.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    llm_with_structure = llm.with_structured_output(schema, include_raw=True)
    # Prompt to capture tool parameters
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert in machine learning. Your task is to extract the parameters of an instruction based on its intention.
            
            ### INTENTION PARAMETERS DEFINITION:
            
            {schema_json}            
            
            ### POSSIBLE PARAMETERS TYPES:
            - "order_id": integer number with 9 digits. 
            - "product_id": integer number with 6 digits.
            - "date": string with format dd/mm/yyyy.
            
            ### ADDTIONAL INFORMATION (to complete all parameters use them if needed):
            Order ID: {order_id}
            Product ID: {product_id}
            
            ### RESTRICTIONS
            If a parameter is not available or not found or does not match type, return "None" in all cases.
            If you cannot copy the data in the instruction, return "None" in all cases.
            Do not invent values. Do not infere any information. In case of doubt, retunr "None".
           
            Instruction: {instruction}
            Intention: {intention}
        """),
        ("human", "Analyze the instruction.")
    ])

    try:
        result = (prompt | llm_with_structure).invoke({"schema_json": schema_json,
                                                       "instruction": instruction,
                                                       "intention": intention,
                                                       "order_id": order_id,
                                                       "product_id": product_id})
        tool_parameters = result["parsed"].model_dump()
        tool_parameters.pop("reasoning", None)
        tool_parameters["order_id"] = order_id
        tool_parameters["product_id"] = product_id
        if "None" in tool_parameters.values():
            error_descr = "Unknown parameter"
        total_tokens = result["raw"].usage_metadata["total_tokens"]

    except Exception as e:
        logger.info(f"CHOOSE TOOL NODE - Exception invoking model: {e}")
        print(f"Error: {e}")
        intention = "Unknown"
        parameters = None
        error_descr = "Unknown intention"

    logger.info(f"CHOOSE TOOL NODE - Quitting")

    return {
        "messages": [AIMessage(content=f"Node choose tool")],
        "tool_params": tool_parameters,
        "params_schema": intention,
        "chosen_tool": state["intention"],
        "total_tokens": total_tokens,
        "error_descr": error_descr,
        "node_name": "HUMAN FEEDBACK"
    }

def node_contact_human(state: State) -> dict:
    """
    Node to build a human readable message requesting for feedback when no tool is selected.
    """

    # Build a human readable message with the conflict
    logger.info(f"HUMAN FEEDBACK NODE - Starting")
    logger.info(f"HUMAN FEEDBACK NODE - Feedback {state["error_descr"]}")
    logger.info(f"HUMAN FEEDBACK NODE - Quitting")

    return {
        "messages": [AIMessage(content=f"Node contact human")],
        "node_name": "HUMAN CONTACT",
        "chosen_tool": None
    }

def tool_node(state: State) -> dict:
    """
    Test graph flow
    """
    logger.info(f"TOOL EXECUTION - Starting")

    tool = tools[state["intention"]]
    result = tool.invoke(state["tool_params"])

    return {
        "messages": [AIMessage(content=f"Test tool_node")],
        "error_descr": result["error_descr"],
        "chosen_tool": result["chosen_tool"],
        "node_name": "TOOL EXECUTION"
    }
#-----------------------------------------------------------------------------------------------------------------------
# Block 6 - Conditional routing
#-----------------------------------------------------------------------------------------------------------------------

def check_intention(state: State) -> Literal["choose_tool", "contact_human"]:
    """
    Checks if intention is clear to avoid calling the model unnecesarily
    """
    logger.info(f"EDGE CHECK CORRECT INTENTION - Starting")
    result = ""
    if len(state["error_descr"])>0:
        result = "contact_human"
    else:
        result = "choose_tool"
    logger.info(f"EDGE CHECK CORRECT INTENTION - Routing to {result}")
    logger.info(f"EDGE CHECK CORRECT INTENTION - Quitting")
    return result

def tool_or_human_calling(state: State) -> Literal["tool_node", "contact_human"]:
    """
    Checks if the model is capable to choose a clear tool with its params. If yes, moves to Tool node.
    If not, move to contact human node.
    """

    # We can go to the tool_node if:
    # 1. There is a clear intention
    # 2. All params are available
    # 3. Confidence is above threshold
    logger.info(f"EDGE CHECK CORRECT TOOL - Starting")
    result = "tool_node"
    if state["intention"].lower() == "unknown":
        result = "contact_human"
    if len(state["error_descr"])>0:
        result = "contact_human"

    logger.info(f"EDGE CHECK CORRECT TOOL - Routing to {result}")
    logger.info(f"EDGE CHECK CORRECT TOOL - Quitting")

    return result

def check_correct_execution(state: State) -> Literal["contact_human", "end"]:
    """
    Routes to End if the tool is executed correctly.
    Routes to give feedback to humans in any other case.
    """
    logger.info(f"EDGE CHECK CORRECT EXECUTION- Starting")
    result = ""
    if len(state["error_descr"])>0:
        result = "contact_human"
    else:
        result = "end"
    logger.info(f"EDGE CHECK CORRECT EXECUTION - Routing to {result}")
    logger.info(f"EDGE CHECK CORRECT EXECUTION - Quitting")
    return result


#-----------------------------------------------------------------------------------------------------------------------
# Block 7 - Build the graph
#-----------------------------------------------------------------------------------------------------------------------

builder = StateGraph(State)

# Add nodes
builder.add_node("read_instruction", node_read_instruction)
builder.add_node("choose_tool", node_choose_tool)
builder.add_node("tool_node", tool_node)
builder.add_node("contact_human", node_contact_human)

# Flow definition
builder.add_edge(START, "read_instruction")
builder.add_conditional_edges(
    "read_instruction",
    check_intention,
    {
        "choose_tool": "choose_tool",
        "contact_human": "contact_human"
    }
)
builder.add_conditional_edges(
    "choose_tool",
    tool_or_human_calling,
    {
        "tool_node": "tool_node",
        "contact_human": "contact_human"
    }
)
builder.add_conditional_edges(
    "tool_node",
    check_correct_execution,
    {
        "contact_human": "contact_human",
        "end": END
    }
)
builder.add_edge("tool_node", END)
builder.add_edge("contact_human", END)

graph = builder.compile()

#-----------------------------------------------------------------------------------------------------------------------
# Block 8 - Run an Example
#-----------------------------------------------------------------------------------------------------------------------
def print_graph_structure(graph: StateGraph) -> None:
    print("-" * 70)
    print("GRAPH FLOW")
    print("-" * 70)
    print(graph.get_graph().draw_ascii())

def execute_instruction(graph: StateGraph, instruction: dict) -> None:
    #print("\n" + "="*70)
    #print(f"INSTRUCTION: {instruction['instruction']} / CUSTOMER GROUP: {instruction['customer_group']}")
    #print("-"*70)

    initial_state = {
        "messages": HumanMessage(content="Processing instruction"),
        "instruction": instruction["instruction"],
        "order_id": instruction["order_id"],
        "product_id": instruction["product_id"],
        "customer_group": instruction["customer_group"],
        "chosen_tool": None,
        "tool_confidence": None,
        "tool_params": None,
        "total_tokens": 0
    }

    timer = time.perf_counter()
    result = graph.invoke(initial_state)
    timer = time.perf_counter() - timer
    hit = str(result["chosen_tool"]) == str(instruction["correct_answer"])

    return {
        "hit": hit,
        "tokens": result["total_tokens"],
        "tool": result["chosen_tool"],
        "time": timer,
        "chosen_tool": result["chosen_tool"]
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
        "instruction": "Release order",
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
        with gr.Row(equal_height=True):
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
    print(f"Total Model Time: {time_model_invoking:.1f}s - Average: {time_model_invoking/total_instructions:.1f}s")
    print(f"Total test time: {timer:.1f}s  Average: {timer/total_instructions:.1f}s ")

#print_graph_structure(graph)
#extended_test(graph, 1, instructions)
#single_case_execution(graph)
