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

# Initialize the AI21 Jamba client
client = AI21Client(api_key=("E13ddoghRLczpStIvPDYIquXdUpGbsqs"))

# Define the tools to interact with Jamba
def get_dti_dataset():
    group = scdti_group.SCDTIGroup()
    train_val = group.get_train_valid_split()
    train = train_val["train"]
    val = train_val["val"]
    test = group.get_test()["test"]
    df = pd.concat([train, val, test], axis=0, ignore_index=True)
    df["protein"] = df["name"]
    df["cell"] = df["cell_type_label"]
    return df

def get_ctspec_protein_embed(cell, protein):
    df = get_embedddings()
    result = df[(df['cell'] == cell) & (df['protein'] == protein)]
    return result

def is_target(cell, protein, disease):
    # Retrieve the dataset
    df = get_dti_dataset()
    
    # Filter for the specific cell, protein, and disease
    result = df[(df['cell_type_label'] == cell) & (df['name'] == protein) & (df['disease'] == disease)]
    return result["y"] == 1
    

def get_embedddings():
    pinnacle = PINNACLE()
    embeds = pinnacle.get_embeds()
    assert isinstance(embeds, DataFrame)
    assert len(embeds) > 0, "PINNACLE embeds is empty"
    keys = pinnacle.get_keys()
    assert isinstance(keys, DataFrame)
    assert len(keys) > 0, "PINNACLE keys is empty"
    assert len(keys) == len(embeds), "{} vs {}".format(len(keys), len(embeds))
    all_entries = embeds.index
    prots = [x.split("--")[0] for x in all_entries]  # proteins are entry 0
    cells = [x.split("--")[1] for x in all_entries]  # cells are entry 1
    embeds["protein"] = prots
    embeds["cell"] = cells
    return embeds

def get_cell_types_for_ra():
    df = get_dti_dataset()
    return df[df["disease"].lower() == "ra"]["cell"].unique()

def get_cell_types_for_ibd():
    df = get_dti_dataset()
    return df[df["disease"].lower() == "ibd"]["cell"].unique()

# # Tool definitions for AI21
# def get_dti_dataset_tool():
#     return ToolDefinition(
#         type="function",
#         function=FunctionToolDefinition(
#             name="get_dti_dataset",
#             description="Get the Drug Target Interaction (DTI) dataset.",
#             parameters=ToolParameters(
#                 type="object",
#                 properties={},
#                 required=[]
#             )
#         )
#     )

# def get_ctspec_protein_embed_tool():
#     return ToolDefinition(
#         type="function",
#         function=FunctionToolDefinition(
#             name="get_ctspec_protein_embed",
#             description="Get cell-type specific protein embedding.",
#             parameters=ToolParameters(
#                 type="object",
#                 properties={
#                     "cell": {"type": "string", "description": "Cell type"},
#                     "protein": {"type": "string", "description": "Protein name"}
#                 },
#                 required=["cell", "protein"]
#             )
#         )
#     )

# def is_target_tool():
#     return ToolDefinition(
#         type="function",
#         function=FunctionToolDefinition(
#             name="is_target",
#             description="Check if a protein is a drug target in a specific cell type and disease. ",
#             parameters=ToolParameters(
#                 type="object",
#                 properties={
#                     "cell": {"type": "string", "description": "Cell type"},
#                     "protein": {"type": "string", "description": "Protein name"},
#                     "disease": {"type": "string", "description": "Disease name"}
#                 },
#                 required=["cell", "protein", "disease"]
#             )
#         )
#     )

# # Register the tools with Jamba
# tools = [
#     # get_dti_dataset_tool(),
#     # get_ctspec_protein_embed_tool(),
#     is_target_tool()
# ]

# messages = [
#     ChatMessage(
#         role="system",
#         content="You are a professional biologist and therapeutics scientist. Always use the is_target tool first.  Answer the questions to the best of your ability."),
    
#     ChatMessage(role="user", content="Is STK38 a target for IBD?"),
# ]


# # Process the tool calls for the AI21 API
# def process_tool_calls(assistant_message):
#     tool_call_id_to_result = {}
#     tool_calls = assistant_message.tool_calls
    
#     if tool_calls:
#         for tool_call in tool_calls:
#             # Handle the is_target tool call
#             if tool_call.function.name == "is_target":
#                 func_arguments = json.loads(tool_call.function.arguments)
#                 cell = func_arguments.get("cell")
#                 protein = func_arguments.get("protein")
#                 disease = func_arguments.get("disease")
#                 if cell and protein and disease:
#                     result = is_target(cell, protein, disease)
#                     tool_call_id_to_result[tool_call.id] = result
#                 else:
#                     print(f"Got unexpected arguments in function call - {func_arguments}")
                    
#             # Handle the get_ctspec_protein_embed tool call
#             elif tool_call.function.name == "get_ctspec_protein_embed":
#                 func_arguments = json.loads(tool_call.function.arguments)
#                 cell = func_arguments.get("cell")
#                 protein = func_arguments.get("protein")
#                 if cell and protein:
#                     result = get_ctspec_protein_embed(cell, protein)
#                     tool_call_id_to_result[tool_call.id] = result
#                 else:
#                     print(f"Got unexpected arguments in function call - {func_arguments}")
                    
#             else:
#                 print(f"Unexpected tool call found - {tool_call.function.name}")
                
#     return tool_call_id_to_result

# # Initial response
# response = client.chat.completions.create(messages=messages, model="jamba-1.5-mini", tools=tools)
# assistant_message = response.choices[0].message
# messages.append(assistant_message)
# tool_call_id_to_result = process_tool_calls(assistant_message)


# # Add tool results to messages
# for tool_id_called, result in tool_call_id_to_result.items():
#     tool_message = ToolMessage(role="tool", tool_call_id=tool_id_called, content=str(result))
#     messages.append(tool_message)

# # # Final response
# response = client.chat.completions.create(messages=messages, model="jamba-1.5-mini", tools=tools)
# final_response = response.choices[0].message.content
# print(final_response)

# print(response)

