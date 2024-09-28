
from openai import OpenAI
import os, time, torch, json
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
REFORM_TEMPLATE = """
Reform the UserQuery into the following format, return only in required format:

<pick one from negative genes, positive genes or both positive and negative genes> in <celltype name in lowercase> during <disease EFO number in uppercase>

---

UserQuery: {question}
"""

CLEANER_PROMPT = """Clean the user query, and reformulate it to make it clearer and actionable. Only return the cleaned query.
User query: {}
"""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query_cleaner(query):
    msg = add_message(CLEANER_PROMPT.format(query))
    return chat_completion(msg)

def add_message(content, role='user', history={}):
    if len(history) == 0:
        return [{"role": role, "content": content}]
    else:
        history.append({"role": role, "content": content})
    return history

def chat_completion(messages):
    response = client.chat.completions.create(
        model='gpt-4',
        # messages=[
        #     {'role': 'user', 'content': formatted_query},
        #     {'role': 'assistant', 'content': history}
        # ],
        messages=messages,
        temperature=1.0,
        top_p=1,
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    return response.choices[0].message.content

def format_query(response_text, sources=None):
    if sources is None:
        formatted_response = f"{bcolors.BOLD}[PINNACLE🗻]{bcolors.ENDC}: {response_text}\n"
    else:
        sources = '\n' + '\n'.join(sources)
        formatted_response = f"{bcolors.BOLD}[PINNACLE🗻]{bcolors.ENDC}: {response_text}\n\nReferences📚: {sources}"
    print('\n---\n')
    print(formatted_response)
    print('\n---\n')
    
def query_reform(query_text):
    query_text = REFORM_TEMPLATE.format(question=query_text)
    query_text = add_message(query_text)
    return chat_completion(query_text)

def form_ppi_embed_dict(celltype_ppi_embed, celltype_dict, celltype_protein_dict):
    # each node(gene) has a vector representation dim: (128,)
    ppi_embed_dict = {}
    for celltype, index in celltype_dict.items():
        cell_embed_dict = {}
        cell_embed = celltype_ppi_embed[index]
        for i, gene in enumerate(celltype_protein_dict[celltype]):
            gene_embed = cell_embed[i, :]
            cell_embed_dict[gene] = gene_embed
            # print(f"[pinnacle]: {celltype} - {gene} - {gene_embed.shape}")
        celltype = celltype.replace(" ", "_")
        ppi_embed_dict[celltype] = cell_embed_dict
        # print(f"[pinnacle]: {celltype} - {len(cell_embed_dict)}")
    return ppi_embed_dict

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