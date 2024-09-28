import json
import torch
import pandas as pd
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ToolMessage
from ai21.models.chat.function_tool_definition import FunctionToolDefinition
from ai21.models.chat.tool_defintions import ToolDefinition
from ai21.models.chat.tool_parameters import ToolParameters


# Define the prompts
REFORM_TEMPLATE = """
Reform the UserQuery into the following format, return only in required format:

<pick one from negative genes, positive genes or both positive and negative genes> in <celltype name in lowercase> during <disease EFO number in uppercase>

---

UserQuery: {question}
"""

CLEANER_PROMPT = """Clean the user query, and reformulate it to make it clearer and actionable. Only return the cleaned query.
User query: {}
"""

# Function to add a message to the history
def add_message(content, role='user', history={}):
    if len(history) == 0:
        return [{"role": role, "content": content}]
    else:
        history.append({"role": role, "content": content})
    return history

# Function to complete chat
def chat_completion(messages):
    response = client.chat.completions.create(
        messages=messages,
        model="jamba-1.5-mini",  # Model selected for Jamba
        temperature=1.0
    )
    return response.choices[0].message.content

# Query Cleaner Tool Definition
def query_cleaner_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="query_cleaner",
            description="Clean the user query and reformulate it to make it clearer and actionable",
            parameters=ToolParameters(
                type="object",
                properties={
                    "query": {
                        "type": "string",
                        "description": "The query that needs to be cleaned"
                    }
                },
                required=["query"]
            )
        )
    )

# Query Cleaner Function
def query_cleaner(query):
    msg = add_message(CLEANER_PROMPT.format(query))
    return chat_completion(msg)

# Query Reform Tool Definition
def query_reform_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="query_reform",
            description="Reformulate the query based on a specific pattern",
            parameters=ToolParameters(
                type="object",
                properties={
                    "query_text": {
                        "type": "string",
                        "description": "The query text that needs to be reformulated"
                    }
                },
                required=["query_text"]
            )
        )
    )

# Query Reform Function
def query_reform(query_text):
    query_text = REFORM_TEMPLATE.format(question=query_text)
    query_text = add_message(query_text)
    return chat_completion(query_text)

# PPI Embed Dict Tool Definition
def form_ppi_embed_dict_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="form_ppi_embed_dict",
            description="Form a dictionary of PPI embeddings for cell types and proteins",
            parameters=ToolParameters(
                type="object",
                properties={
                    "celltype_ppi_embed": {
                        "type": "array",
                        "description": "The PPI embeddings for the cell types"
                    },
                    "celltype_dict": {
                        "type": "object",
                        "description": "Dictionary mapping cell types to their indices"
                    },
                    "celltype_protein_dict": {
                        "type": "object",
                        "description": "Dictionary mapping cell types to their protein lists"
                    }
                },
                required=["celltype_ppi_embed", "celltype_dict", "celltype_protein_dict"]
            )
        )
    )

# PPI Embed Dict Function
def form_ppi_embed_dict(celltype_ppi_embed, celltype_dict, celltype_protein_dict):
    ppi_embed_dict = {}
    for celltype, index in celltype_dict.items():
        cell_embed_dict = {}
        cell_embed = celltype_ppi_embed[index]
        for i, gene in enumerate(celltype_protein_dict[celltype]):
            gene_embed = cell_embed[i, :]
            cell_embed_dict[gene] = gene_embed
        celltype = celltype.replace(" ", "_")
        ppi_embed_dict[celltype] = cell_embed_dict
    return ppi_embed_dict

# Load Embed Tool Definition
def load_embed_only_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="load_embed_only",
            description="Load embeddings and associated labels for specific cell types and proteins",
            parameters=ToolParameters(
                type="object",
                properties={
                    "embed_path": {
                        "type": "string",
                        "description": "Path to the embedding file"
                    },
                    "labels_path": {
                        "type": "string",
                        "description": "Path to the labels file"
                    }
                },
                required=["embed_path", "labels_path"]
            )
        )
    )

# Load Embed Function
def load_embed_only(embed_path: str, labels_path: str):
    embed = torch.load(embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    celltype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(celltypes)}
    assert len(celltype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p)
        protein_celltypes.append(c)

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(celltype_protein_dict) > 0
    return embed, celltype_dict, celltype_protein_dict

# Format Query Tool Definition
def format_query_tool():
    return ToolDefinition(
        type="function",
        function=FunctionToolDefinition(
            name="format_query",
            description="Format a response with optional sources",
            parameters=ToolParameters(
                type="object",
                properties={
                    "response_text": {
                        "type": "string",
                        "description": "The main response text"
                    },
                    "sources": {
                        "type": "array",
                        "description": "Optional list of sources",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                required=["response_text"]
            )
        )
    )

# Format Query Function
def format_query(response_text, sources=None):
    if sources is None:
        formatted_response = f"[PINNACLE🗻]: {response_text}\n"
    else:
        sources = '\n' + '\n'.join(sources)
        formatted_response = f"[PINNACLE🗻]: {response_text}\n\nReferences📚: {sources}"
    print('\n---\n')
    print(formatted_response)
    print('\n---\n')

# Define and load the tools
tools = [
    query_cleaner_tool(),
    query_reform_tool(),
    form_ppi_embed_dict_tool(),
    load_embed_only_tool(),
    format_query_tool()
]

