# from openai import OpenAI
# from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# from retrival_protein_regulation import load_documents, split_text, save_to_chroma

from cell_annotation import query_celltype_ppi
from cell_annotation import query_cleaner
from gpt_utils import chat_completion
from gpt_utils import format_query
from gpt_utils import query_reform
from gpt_utils import load_embed_only
from gpt_utils import form_ppi_embed_dict
from gpt_utils import add_message

from disease_utils import disease_conditioned_info

import os
import argparse
import warnings
import torch

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question and use following context as reference:

{context}

---

Question: {question}
"""


TASK_VERIFY_TEMPLATE = """
Based on the UserQuery, verify if it belongs to one of the following tasks and return in corresponding output format:

1. Ask question/query relate to genes/celltypes in a specific disease
Output: provide the disease official name in format '[pinnacle-disease]:<disease official name>'. 

2. Asking question/query about genes/celltypes in regular state
Output: '[pinnacle]:<celltype official name>'

3. General question
Output: '[general]'

---

UserQuery: {question}
"""

GENERAL_RESPONSE = """ You are a professional biologist and therapeutics scientist. When user asking for assitant, please respond concisely and truthfully. Otherwise, chat with user in a kind and informative manner.

---

User: {query_text}
"""


def retrival_context(query_text, llm_database_path, efo_id):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=llm_database_path, embedding_function=embedding_function)
    # Search the DB.
    query_text = query_reform(query_text)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # print(results)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text+f'({efo_id})')
    # print(prompt)
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response_text = model.invoke(prompt).content
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    format_query(response_text, sources)

if __name__ == "__main__":
    
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_path", type=str, default="../PrismDB/pinnacle_embeds/pinnacle_protein_embed.pth")
    parser.add_argument("--embed_labels_path", type=str, default="../PrismDB/pinnacle_embeds/pinnacle_labels_dict.txt")
    parser.add_argument("--rag_folder", type=str, default="../PrismDB/therapeutic_target_data/", help="data folder for disease specific gene expression data")
    parser.add_argument("--llm_database_path", type=str, default="../PrismDB/chroma", help="User query text")
    parser.add_argument("--ppi_embed_path", type=str, default="../PrismDB/disease_conditioned/ppi_embed_dict.pth")
    args = parser.parse_args()
    
    rag_folder = args.rag_folder
    llm_database_path = args.llm_database_path
    
    if not os.path.exists(rag_folder):
        os.makedirs(rag_folder)
    
    if not os.path.exists(args.ppi_embed_path):
        ppi_embed, celltype_dict, celltype_protein_dict = load_embed_only(args.embed_path, args.embed_labels_path)
        ppi_embed_dict = form_ppi_embed_dict(ppi_embed, celltype_dict, celltype_protein_dict)
    else:
        ppi_embed_dict = torch.load(args.ppi_embed_path)
        
    disease_set, history = {}, []
    while True:
        # Get user query & verify the task.
        disease_evidence_flag = False
        query_text = input("[User]: ")
        query_text = query_cleaner(query_text)
        instruction = TASK_VERIFY_TEMPLATE.format(question=query_text)
        instruction = add_message(instruction)
        content = chat_completion(instruction)
        
        # Collect extra infor needed for disease conditioned query.
        try:
            if content.split(":")[0] == "[pinnacle-disease]":
                disease_name = content.split(':')[1]
                efo_number, positive_proteins, negative_proteins = disease_conditioned_info(disease_name)
                disease_set = {'disease_name':disease_name, 
                            'efo_number':efo_number, 
                            'positive_proteins':positive_proteins, 
                            'negative_proteins': negative_proteins}
        except Exception as e:
            print(e)
            print("[pinnacle] Can not find relative druggable genes from ChEMBL. Answering using general knowledge...")
            disease_evidence_flag = True

        if "pinnacle" in content.split(":")[0] and not disease_evidence_flag:
            print('check')
            final_ans = query_celltype_ppi(query_text, history, ppi_embed_dict, **disease_set)
        else:
            general_msg = add_message(GENERAL_RESPONSE.format(query_text=query_text), role='user', history=history)
            final_ans = chat_completion(general_msg)

        format_query(final_ans)
        history = add_message(query_text, history=history)
        history = add_message(final_ans, role='assistant', history=history)
        
