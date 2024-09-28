from setup import create_parser, get_hparams, setup_paths
from read_data import load_data
from mdutils.mdutils import MdUtils

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

import numpy as np

import random
import wandb
import json
import torch
import shutil
import os

DATA_PATH = '../PrismDB/theraputic_llm/'
CHROMA_PATH = "../PrismDB/chroma"

def emb_to_naturaldb(positive_proteins, negative_proteins, celltype_dict, task_name):
    """
    Given a tensor of embeddings, return the natural database representation
    """
    # Load data
    pos_embed = []
    neg_embed = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    task_data_path = DATA_PATH
    # Additional Markdown syntax..
    print(f'Generating markdown files for {task_name}...')
    pos_md = MdUtils(file_name=task_data_path + f'positive_proteins_{task_name}',title=f'Genes/Proteins with positive regulation/expression during {task_name}')
    neg_md = MdUtils(file_name=task_data_path + f'negative_proteins_{task_name}',title=f'Genes/Proteins with negative regulation/expression during {task_name}')
    
    # Generate data for split 
    for celltype in celltype_dict: 
        celltype = celltype.replace(' ', '_')
        if celltype not in positive_proteins: continue

        pos_prots = positive_proteins[celltype]
        neg_prots = negative_proteins[celltype]
        # pos_md.new_header(level=1, title=f'Positive gene/protein expressed in {celltype} during {task_name}')
        # neg_md.new_header(level=1, title=f'Negative gene/protein expressed in {celltype} during {task_name}')
        pos_paragraph, neg_paragraph = ', '.join(pos_prots), ', '.join(neg_prots)
    
        md_celltype = celltype.replace('_', ' ')
        pos_wrapper = f'positive genes in {md_celltype} during {task_name} including: '
        neg_wrapper = f'negative genes in {md_celltype} during {task_name} including: '
        wrapper_end = f'. That is all for {md_celltype} during {task_name}.'
        pos_paragraph = ' '.join([pos_wrapper, pos_paragraph, wrapper_end])
        neg_paragraph = ' '.join([neg_wrapper, neg_paragraph, wrapper_end])
        # print(pos_paragraph)
        
        pos_md.new_paragraph(pos_paragraph)
        neg_md.new_paragraph(neg_paragraph)
        pos_md.new_line()
        neg_md.new_line()
    pos_md.create_md_file()
    neg_md.create_md_file()
    print(f'[pinnacle] Saving markdown files to {task_data_path}...')
    return pos_embed, neg_embed

def load_documents():
    task_data_path = DATA_PATH
    loader = DirectoryLoader(task_data_path, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
def main(args):
    _, _, _, embed_path, labels_path = setup_paths(args)
    print(args.positive_proteins_prefix)
    _, celltype_dict, _, positive_proteins, negative_proteins, _ = load_data(embed_path, labels_path, args.positive_proteins_prefix, args.negative_proteins_prefix, None)
    emb_to_naturaldb(positive_proteins, negative_proteins, celltype_dict, args.task_name)
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    
if __name__ == '__main__':
    args = create_parser()
    main(args)
    # usage example
    # python retrival_protein_regulation.py \
    # --task_name EFO_0000616 \
    # --embeddings_dir ../data/pinnacle_embeds/ \
    # --positive_proteins_prefix ../data/therapeutic_target_task/positive_proteins_EFO_0000616 \
    # --negative_proteins_prefix ../data/therapeutic_target_task/negative_proteins_EFO_0000616  