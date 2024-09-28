import glob
import os
import json
import networkx as nx
import pickle as pkl

import torch
import warnings
import pickle

from gpt_utils import chat_completion
from gpt_utils import query_cleaner
from gpt_utils import add_message
from subprocess import PIPE, Popen
from copy import copy
# from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# new imports
import pinnacle
from tdc.resource.pinnacle import PINNACLE

# global variables
EFO_NUM = None
POSITIVE_PROTEINS = None
NEGATIVE_PROTEINS = None
PPI_EMBED_DICT = {}

DISEASE_INFO_IMPORT = """To use POSITIVE_PROTEINS and NEGATIVE_PROTEINS, remember to add the following at the beginning of the code:
import pickle as pkl
with open('{efo_number}_objs.pkl', "rb") as f:
    POSITIVE_PROTEINS, NEGATIVE_PROTEINS = pkl.load(f)
"""

DISEASE_SPECIFIC_PROMPT = """3. POSITIVE_PROTEINS - dictionary of proteins that are positive druggable target for {disease_name} collected from CHEMBL and DrugBank
    Example:
        POSITIVE_PROTEINS['b_cell'] returns the list of proteins that are positive druggable target in B cells for {disease_name} 
    
    4. NEGATIVE_PROTEINS - dictionary of proteins that are negative drug target for {disease_name} collected from CHEMBL and DrugBank
    Example:
        NEGATIVE_PROTEINS['b_cell'] returns the list of proteins that are negative druggable target in B cells for {disease_name}
"""

TASK_PROMPT = """Write python code (might use the following PPI dictionary as support) to answer the user query. Return only in code.
    Avaliable PPI dictionaries:
    1. ppi_layers - dictionary of PPI networks, where the key is the cell type context and the value is the networkx graph
    Example: 
        ppi_layers['b_cell'] returns the PPI network for B cells in the form of a networkx graph
    
    2. PPI_EMBED_DICT - dictionary of pretrained PPI embeddings in healthy individual(from PINNACLE https://github.com/mims-harvard/PINNACLE), where the key is the cell type context and the value is a dictionary of gene embeddings
    Example: 
        PPI_EMBED_DICT['plasma_cell'] returns the gene embedding dictionary for plasma cells
        PPI_EMBED_DICT['plasma_cell']['CD19'] returns the embedding for CD19 in plasma cells, which is a 128-dimensional tensor, torch.Size([128])
    {disease_context}
    ---
    Notice:
    1. To use ppi_layers, remember to import read_ppi and add the following at the beginning of the code:
    from cell_annotation import read_ppi
    ppi_layers = read_ppi("../PrismDB/networks/ppi_edgelists/")
    2. To use PPI_EMBED_DICT, remember to add the following at the beginning of the code:
    import torch
    PPI_EMBED_DICT = torch.load("{ppi_embed_dict_path}")
    {disease_dict_import}
    
    3. All methods and functions from Python library "Networkx" can be used. Remember to import networkx at the beginning of the code.
    4. It is encouraged to import any suitable external python packages for solving the task.
    4. The celltype-specific gene embedding from PPI_EMBED_DICT is a 128-dimensional tensor, torch.Size([128])
    5. The embedding for a particular cell type's gene can be accessed by PPI_EMBED_DICT[celltype][protein_name]
    6. The embedding for a particular cell type's gene learned from node-level attention, indicating the importance of the gene to its neighboring genes in the celltype specific PPI network.
    7. The embedding weight for a particular cell type's gene from PPI_EMBED_DICT do not contain info about the gene's expression level in the cell type.
    8. Cell type context is the key of the ppi_layers dictionary, and is write in lower case and with space replaced by underscore. For example, 'B cell' should be written as 'b_cell'
    9. Cell type context should be represented as singular form. For example, 'B cells' should be written as 'b_cell'.
    10 When encountering cell types with descriptive phrases, preserve the phrases markes like '-' and ','. For example, 'cd4-positive helper t cell' should be written as 'cd4-positive_helper_t_cell', 'cd4-positive, alpha-beta memory t cell' should be written as 'cd4-positive,_alpha-beta_memory_t_cell'
    11. Dont forget to import nectworkx and other necessary library at the beginning of the code if needed.
    12. The code should printout the answer to the user query.
    13. Always wrap the code using ```python at the beginning and ``` at the end
    14. For complex tasks, think about breaking down the task into smaller subtasks and solve them one by one.
    15. If you can not directly find the celltype in the PPI_EMBED_DICT or ppi_layers, you can try to find celltypes in the PPI_EMBED_DICT or ppi_layers that belong to the same cell lineage or cell type family.
    ---
    {debug_info}
    User query: {query}
    
    """

DEBUG_PROMPT = """ Here is a answer from your previous attempt: {}
Comments: {}
Please improve the code based on comments, try again.
---
"""


ANSWER_PROMPT = """As an expert in genetics, bioinformatics, and biology, your role is to provide the most accurate response to user inquiries, utilizing relevant information from analytical results when necessary. Please ensure your responses are clear, succinct, and truthful. Remember to focus solely on responding to the question asked.
---
User query: {}
Analyzed reasult: {}
---
Code used for the analysis:
{}
---
Note: 
1. If you confident with the analyzed result, you can use it as support to answer the user query.
2. If you are not confident with the analyzed result, ignore it and answer the user query based on your own knowledge.
3. If you wish to use calculation scores from the analyzed result, explain the scores in the answer and round the scores to 2 decimal places and .
4. Make sure the answer is elegant, professiona and well-structured.
"""

QUALITY_CHECK = """Here is the code generated by GPT to help answer the user query:
{query}
---
Code generated by GPT:
{responsed_code}
---
Here is the output of the code:
---
{code_output}
---
Suppose the source file mentioned in the generated code is provided and contains necessary info (form using PINNACLE, ChEMBL, DrugBank and OSL). 
Do you agree that the code and its generated output are both correct and suitable? Can you point to any evidence reinforcing the credibility of this output? What reasoning underlies your stance?
Output: (yes/no) - (Provide your rationale in 100 words or less, mentioning if any metrics, software packages, or potential modifications could enhance the code if required)
---
Notes: 
1. Use " - " as a separator between the yes/no and the reason.
2. If any metrics, tools or corrective measures are referenced in your justification, please make sure to clearly specify them.
"""

# warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
# warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

def get_ppi():
    return pinnacle.get_ppi()

def get_context_specific_ppis():
    ppi = get_ppi()
    pins = PINNACLE()
    dti_df = pins.get_dti_dataset()
    cells = dti_df["cell_type_label"].unique()
    out = {}
    for cell in cells:
        ct_prots = dti_df[(dti_df['cell'] == cell)]["protein"].unique()
        df = ppi
        ct_ppi = df[(df['Protein A'].isin(ct_prots)) & (df['Protein B'].isin(ct_prots))]
        ct_ppi["context"] = cell
        out["cell"] = ct_ppi
    return out



def code_quality_check(query, responsed_code, code_output):
    prompt = QUALITY_CHECK.format(query=query, responsed_code=responsed_code, code_output=code_output)
    prompt = add_message(prompt)
    while True:
        try:
            eval_response = chat_completion(prompt)
            decision, reason = eval_response.split("-")
        except Exception as e:
            # print("[pinnacle] Error in code quality check. Retrying...")
            continue
        break
    # print(eval_response)
    # print(decision)
    if "yes" in decision.lower():
        return True, reason.strip()
    return False, reason.strip()
    

def read_embed(ppi_embed_f, mg_embed_f, labels_f):

    # Read embeddings
    ppi_embed = torch.load(ppi_embed_f)
    embeddings = []
    for celltype, x in ppi_embed.items():
        embeddings.append(x)
    embeddings = torch.cat(embeddings)
    print("PPI embeddings", embeddings.shape)

    # Read metagraph embeddings
    mg_embed = torch.load(mg_embed_f)
    print("Meta graph embeddings", mg_embed.shape)

    # Read labels
    with open(labels_f) as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)

    return embeddings, mg_embed, labels_dict
    

def query_celltype_ppi(query, history, ppi_embed_dict: dict, ppi_embed_dict_path = '../data/disease_conditioned', **kwargs):
    print("[pinnacle] Cleaned query: ", query)
    debug_info, disease_context, disease_dict_import = "", "", ""
    ppi_embed_dict_path = os.path.join(ppi_embed_dict_path, 'ppi_embed_dict.pth')
    
    if not os.path.exists(ppi_embed_dict_path):
        print("[pinnacle] Saving ppi_embed_dict...")
        torch.save(ppi_embed_dict, ppi_embed_dict_path)
        
    if len(kwargs) > 0:
        disease_name, efo_number, positive_proteins, negative_proteins = kwargs.values()
        with open(f'{efo_number}_objs.pkl', 'wb') as f:
            pkl.dump([positive_proteins, negative_proteins], f)
        disease_context = DISEASE_SPECIFIC_PROMPT.format(disease_name=disease_name)
        disease_dict_import = DISEASE_INFO_IMPORT.format(efo_number=efo_number)
    
    result = None
    for i in range(5):
        try:
            temp_query = copy(query)
            task_prompt = TASK_PROMPT.format(query=temp_query, 
                                            disease_context=disease_context, 
                                            ppi_embed_dict_path=ppi_embed_dict_path, 
                                            debug_info=debug_info, 
                                            disease_dict_import=disease_dict_import)
            
            msg = add_message(task_prompt)
            responsed_code = chat_completion(msg)
            print("\n=====\n[pinnacle] Attempt - {}:\n=====\n".format(i+1))
            # print(responsed_code)
            responsed_code = responsed_code.split("```python")[1].split("```")[0].strip()
        except Exception as e:
            print("[pinnacle] Error in code formatting. Retrying...")
            continue
        
        print("[pinnacle] Code generation:")
        print('===============\n'+responsed_code+'\n===============\n')
        with open("user_query_task.py", "w") as f:
            f.write("import warnings\n")
            f.write("""warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")\n""")
            f.write(responsed_code)
            
        p = Popen("python user_query_task.py", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if p.returncode and stderr:
            print("[pinnacle] Error in code execution. Retrying...")
            error_msg = stderr.decode("utf-8")
            debug_info = DEBUG_PROMPT.format(responsed_code, error_msg)
            print(f"[pinnacle] error_msg in previous attemp: \n{error_msg}")
            continue
        
        result = stdout.decode("utf-8")
        decision, reason = code_quality_check(query, responsed_code, result)
        if not decision or i < 1:
            debug_info = DEBUG_PROMPT.format(responsed_code, reason)
            # print(f"[pinnacle] Bad code quality, retrying... : \n{reason}")
            continue
        print(f"[pinnacle] Code Decision: Accepted\nReason: {reason}")
        break
    print("[pinnacle] Final result: ", result)
    answer_prompt = ANSWER_PROMPT.format(query, result, responsed_code)
    msg = add_message(answer_prompt, history=history)
    final_ans = chat_completion(msg)
    return final_ans

if __name__ == '__main__':
    # input_f = "../data/pinnacle_embeds/"
    # print("Read in data...")
    # ppi_x, mg_x, labels_dict = read_embed(input_f + "pinnacle_protein_embed.pth", input_f + "pinnacle_mg_embed.pth", input_f + "pinnacle_labels_dict.txt")
    # ppi_layers = read_ppi("../data/networks/ppi_edgelists/")
    # metagraph = nx.read_edgelist("../data/networks/mg_edgelist.txt", delimiter = "\t")
    # for key, ppi in ppi_layers.items():
    #     sorted_nodes = sorted(ppi.degree, key=lambda x: x[1], reverse=True)
    
    # list the top 10 proteins name with the highest expression in connective tissue cell
    query = input("Enter the user query: ")
    query_celltype_ppi(query_cleaner(query))
    # what 
    # Which cell type is most affected by ulcerative colitis?
    # Any genes in <celltype> might be associated with ulcerative colitis?
    # Give me the top 5 genes in goblet cells have strong gene-disease association with ulcerative colitis
    # Compare gene regulation between healthy and ulcerative colitis, give me the top 5 gene pairs that might cause sythetic lethal
    
    