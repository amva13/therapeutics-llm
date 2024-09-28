# import glob
# import os
# import json
# import networkx as nx
# import pickle as pkl

# import torch
# import warnings
# import pickle

# from gpt_utils import chat_completion
# from gpt_utils import query_cleaner
# from gpt_utils import add_message
# from subprocess import PIPE, Popen
# from copy import copy
# from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# new imports
import pinnacle
from tdc.resource.pinnacle import PINNACLE
from tdc.feature_generators.protein_feature_generator import ProteinFeatureGenerator
from tdc.multi_pred import GDA

import pandas as pd

def get_ppi():
    return pinnacle.get_ppi()

def get_context_specific_ppis():
    ppi = get_ppi()
    # pins = PINNACLE()
    dti_df = pinnacle.get_dti_dataset()
    cells = dti_df["cell_type_label"].unique()
    out = {}
    for cell in cells:
        ct_prots = dti_df[(dti_df['cell'] == cell)]["protein"].unique()
        df = ppi
        ct_ppi = df[(df['Protein A'].isin(ct_prots)) & (df['Protein B'].isin(ct_prots))]
        ct_ppi["context"] = cell
        out["cell"] = ct_ppi
    return out



# def code_quality_check(query, responsed_code, code_output):
#     prompt = QUALITY_CHECK.format(query=query, responsed_code=responsed_code, code_output=code_output)
#     prompt = add_message(prompt)
#     while True:
#         try:
#             eval_response = chat_completion(prompt)
#             decision, reason = eval_response.split("-")
#         except Exception as e:
#             # print("[pinnacle] Error in code quality check. Retrying...")
#             continue
#         break
#     # print(eval_response)
#     # print(decision)
#     if "yes" in decision.lower():
#         return True, reason.strip()
#     return False, reason.strip()
    

def read_embed():

    # pins = PINNACLE()
    embeddings = pinnacle.get_embedddings()
    # add labels
    dtis = pinnacle.get_dti_dataset()
    df_merged = pd.merge(embeddings, dtis, on=["cell", "protein"], how="inner")
    labels_df = dtis[["name","cell_type_label", "y"]]
    res = labels_df.set_index(["name", "cell_type_label"])["y"].to_dict()

    return df_merged, None, res  # used to have a metagraph embedding, but that's not needed here?

def get_protein_sequence(gene):
    import time
    import random
    for _ in range(5):
        try:
            return ProteinFeatureGenerator.get_protein_sequence(gene)
        except:
            # backoff
            wait = random.uniform(1,10)
            time.sleep(wait)
    return ""  # empty string if we don't get the sequence

def get_gene_disease_association(gene):
    seq = get_protein_sequence(gene)
    if not seq:
        return -1
    data = GDA(name = 'DisGeNET')
    df = data.get_data()
    gda = df[df["Gene"] == seq]["Y"]
    return gda

def get_cells_in_tissue(tissue):
    mg = pinnacle.get_cell_mg()
    return mg[mg["Tissue"] == tissue]["Cell"].unique()


    

# def query_celltype_ppi(query, history, ppi_embed_dict: dict, ppi_embed_dict_path = '../data/disease_conditioned', **kwargs):
#     print("[pinnacle] Cleaned query: ", query)
#     debug_info, disease_context, disease_dict_import = "", "", ""
#     ppi_embed_dict_path = os.path.join(ppi_embed_dict_path, 'ppi_embed_dict.pth')
    
#     if not os.path.exists(ppi_embed_dict_path):
#         print("[pinnacle] Saving ppi_embed_dict...")
#         torch.save(ppi_embed_dict, ppi_embed_dict_path)
        
#     if len(kwargs) > 0:
#         disease_name, efo_number, positive_proteins, negative_proteins = kwargs.values()
#         with open(f'{efo_number}_objs.pkl', 'wb') as f:
#             pkl.dump([positive_proteins, negative_proteins], f)
#         disease_context = DISEASE_SPECIFIC_PROMPT.format(disease_name=disease_name)
#         disease_dict_import = DISEASE_INFO_IMPORT.format(efo_number=efo_number)
    
#     result = None
#     for i in range(5):
#         try:
#             temp_query = copy(query)
#             task_prompt = TASK_PROMPT.format(query=temp_query, 
#                                             disease_context=disease_context, 
#                                             ppi_embed_dict_path=ppi_embed_dict_path, 
#                                             debug_info=debug_info, 
#                                             disease_dict_import=disease_dict_import)
            
#             msg = add_message(task_prompt)
#             responsed_code = chat_completion(msg)
#             print("\n=====\n[pinnacle] Attempt - {}:\n=====\n".format(i+1))
#             # print(responsed_code)
#             responsed_code = responsed_code.split("```python")[1].split("```")[0].strip()
#         except Exception as e:
#             print("[pinnacle] Error in code formatting. Retrying...")
#             continue
        
#         print("[pinnacle] Code generation:")
#         print('===============\n'+responsed_code+'\n===============\n')
#         with open("user_query_task.py", "w") as f:
#             f.write("import warnings\n")
#             f.write("""warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")\n""")
#             f.write(responsed_code)
            
#         p = Popen("python user_query_task.py", shell=True, stdout=PIPE, stderr=PIPE)
#         stdout, stderr = p.communicate()
#         if p.returncode and stderr:
#             print("[pinnacle] Error in code execution. Retrying...")
#             error_msg = stderr.decode("utf-8")
#             debug_info = DEBUG_PROMPT.format(responsed_code, error_msg)
#             print(f"[pinnacle] error_msg in previous attemp: \n{error_msg}")
#             continue
        
#         result = stdout.decode("utf-8")
#         decision, reason = code_quality_check(query, responsed_code, result)
#         if not decision or i < 1:
#             debug_info = DEBUG_PROMPT.format(responsed_code, reason)
#             # print(f"[pinnacle] Bad code quality, retrying... : \n{reason}")
#             continue
#         print(f"[pinnacle] Code Decision: Accepted\nReason: {reason}")
#         break
#     print("[pinnacle] Final result: ", result)
#     answer_prompt = ANSWER_PROMPT.format(query, result, responsed_code)
#     msg = add_message(answer_prompt, history=history)
#     final_ans = chat_completion(msg)
#     return final_ans

if __name__ == '__main__':
    get_ppi()
    get_context_specific_ppis()
    read_embed()
    get_protein_sequence("STK38")
    get_gene_disease_association("STK38")
    print(get_cells_in_tissue("acinar cell of salivary gland"))
    # input_f = "../data/pinnacle_embeds/"
    # print("Read in data...")
    # ppi_x, mg_x, labels_dict = read_embed(input_f + "pinnacle_protein_embed.pth", input_f + "pinnacle_mg_embed.pth", input_f + "pinnacle_labels_dict.txt")
    # ppi_layers = read_ppi("../data/networks/ppi_edgelists/")
    # metagraph = nx.read_edgelist("../data/networks/mg_edgelist.txt", delimiter = "\t")
    # for key, ppi in ppi_layers.items():
    #     sorted_nodes = sorted(ppi.degree, key=lambda x: x[1], reverse=True)
    
    # list the top 10 proteins name with the highest expression in connective tissue cell
    # query = input("Enter the user query: ")
    # query_celltype_ppi(query_cleaner(query))
    # what 
    # Which cell type is most affected by ulcerative colitis?
    # Any genes in <celltype> might be associated with ulcerative colitis?
    # Give me the top 5 genes in goblet cells have strong gene-disease association with ulcerative colitis
    # Compare gene regulation between healthy and ulcerative colitis, give me the top 5 gene pairs that might cause sythetic lethal
     