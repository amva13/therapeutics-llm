# Import the libraries needed
from tdc.benchmark_group import scdti_group
from tdc.resource.pinnacle import PINNACLE
from pandas import DataFrame
import pandas as pd
import os
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ToolMessage
from ai21.models.chat.function_tool_definition import FunctionToolDefinition
from ai21.models.chat.tool_defintions import ToolDefinition
from ai21.models.chat.tool_parameters import ToolParameters
from pinnacle import is_target, get_ctspec_protein_embed
from primekg import *
from cell_annotation import *
import json

# Initialize the AI21 Jamba client
client = AI21Client(api_key=("E13ddoghRLczpStIvPDYIquXdUpGbsqs"))


# Tool definitions for AI21
def get_dti_dataset_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_dti_dataset",
            description="Get the Drug Target Interaction (DTI) dataset.",
            parameters=ToolParameters(
                type="object",
                properties={},
                required=[]
            )
        )
    )

def get_ctspec_protein_embed_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_ctspec_protein_embed",
            description="Get cell-type specific protein embedding.",
            parameters=ToolParameters(
                type="object",
                properties={
                    "cell": {"type": "string", "description": "Cell type"},
                    "protein": {"type": "string", "description": "Protein name"}
                }
            )
        )
    )

def is_target_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="is_target",
            description="Check if a protein is a drug target in a specific cell type and disease.Returns a 1 if true and 0 if false",
            parameters=ToolParameters(
                type="object",
                properties={
                    "cell": {"type": "string", "description": "Cell type"},
                    "protein": {"type": "string", "description": "Protein name"},
                    "disease": {"type": "string", "description": "Disease name"}
                }
            )
        )
    )


def get_all_drug_evidence_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_all_drug_evidence",
            description="Retrieve all drugs interacting with proteins relevant to a disease.",
            parameters=ToolParameters(
                type="object",
                properties={
                    "disease": {"type": "string", "description": "Disease name"}
                }
            )
        )
    )

def get_all_associated_targets_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_all_associated_targets",
            description="Get all associated protein targets for a disease.",
            parameters=ToolParameters(
                type="object",
                properties={
                    "disease": {"type": "string", "description": "Disease name"}
                }
            )
        )
    )

def get_disease_disease_associations_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_disease_disease_associations",
            description="Retrieve disease-disease associations for a given disease.",
            parameters=ToolParameters(
                type="object",
                properties={
                    "disease": {"type": "string", "description": "Disease name"}
                }
            )
        )
    )

def get_labels_from_evidence_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="get_labels_from_evidence",
            description="Retrieve labels from evidence for a disease.",
            parameters=ToolParameters(
                type="object",
                properties={
                    "disease": {"type": "string", "description": "Disease name"}
                }
            )
        )
    )


# Register the tools with Jamba
tools = [
    get_dti_dataset_tool(),
    get_ctspec_protein_embed_tool(),
    is_target_tool(),
    get_all_drug_evidence_tool(),
    get_all_associated_targets_tool(),
    get_disease_disease_associations_tool(),
    get_labels_from_evidence_tool()
]

messages = [
    ChatMessage(
        role="system",
        content='''
You are a professional biologist and therapeutics scientist. Use the supplied tools to answer the query provided by the user.
Think step by step. You run in a loop of THOUGHT, ACTION, OBSERVATION. 
At the end of the loop you output an ANSWER.
Use THOUGHT to describe your thoughts about the question you have been asked.
Use ACTION to run one of the actions available to you.
OBSERVATION will be the result of running those actions.

The ANSWER should directly and succintly answer the user's original query. You should not write code in the ANSWER, only provide a direct answer to the user's query.

'''),
    
    ChatMessage(role="user", content="Is STK38 a target for IBD?"),
]


# Process the tool calls for the AI21 API
def process_tool_calls(assistant_message):
    tool_call_id_to_result = {}
    tool_calls = assistant_message.tool_calls
    
    if tool_calls:
        for tool_call in tool_calls:
            # Handle the is_target tool call
            if tool_call.function.name == "is_target":
                func_arguments = json.loads(tool_call.function.arguments)
                cell = func_arguments.get("cell")
                protein = func_arguments.get("protein")
                disease = func_arguments.get("disease")
                if cell and protein and disease:
                    result = is_target(cell, protein, disease)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")
                    
            # Handle the get_ctspec_protein_embed tool call
            elif tool_call.function.name == "get_ctspec_protein_embed":
                func_arguments = json.loads(tool_call.function.arguments)
                cell = func_arguments.get("cell")
                protein = func_arguments.get("protein")
                if cell and protein:
                    result = get_ctspec_protein_embed(cell, protein)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")

            # Handle the get_all_drug_evidence tool call
            elif tool_call.function.name == "get_all_drug_evidence":
                func_arguments = json.loads(tool_call.function.arguments)
                disease = func_arguments.get("disease")
                if disease:
                    result = get_all_drug_evidence(disease)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")

            # Handle the get_all_associated_targets tool call
            elif tool_call.function.name == "get_all_associated_targets":
                func_arguments = json.loads(tool_call.function.arguments)
                disease = func_arguments.get("disease")
                if disease:
                    result = get_all_associated_targets(disease)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")

            # Handle the get_disease_disease_associations tool call
            elif tool_call.function.name == "get_disease_disease_associations":
                func_arguments = json.loads(tool_call.function.arguments)
                disease = func_arguments.get("disease")
                if disease:
                    result = get_disease_disease_associations(disease)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")

            # Handle the get_labels_from_evidence tool call
            elif tool_call.function.name == "get_labels_from_evidence":
                func_arguments = json.loads(tool_call.function.arguments)
                disease = func_arguments.get("disease")
                if disease:
                    result = get_labels_from_evidence(disease)
                    tool_call_id_to_result[tool_call.id] = result
                else:
                    print(f"Got unexpected arguments in function call - {func_arguments}")

            else:
                print(f"Unexpected tool call found - {tool_call.function.name}")
                
    return tool_call_id_to_result

# Initial response
response = client.chat.completions.create(messages=messages, model="jamba-1.5-mini", tools=tools)
assistant_message = response.choices[0].message
messages.append(assistant_message)
tool_call_id_to_result = process_tool_calls(assistant_message)


# Add tool results to messages
for tool_id_called, result in tool_call_id_to_result.items():
    tool_message = ToolMessage(role="tool", tool_call_id=tool_id_called, content=str(result))
    messages.append(tool_message)

# # Final response
response = client.chat.completions.create(messages=messages, model="jamba-1.5-mini", tools=tools)
final_response = response.choices[0].message.content
print(final_response)

print(response)