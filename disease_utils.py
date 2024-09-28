from gpt_utils import chat_completion, add_message
from read_data import read_labels_from_evidence

import json
import requests
import os

# Set base URL of GraphQL API endpoint
base_url = "https://api.platform.opentargets.org/api/v4/graphql"

open_target_query_string = """
    query target($diseaseName: String!){
        search(queryString: $diseaseName){
        hits{
            id
            entity
            category
            name
            description
        }
    }
}
"""

QUERY_DISEASEID = """
Please provide the disease ID of {disease_name} from OpenTargets using the format: 'DiseaseID of <disease>: <diseaseID>'.
---
Here are related disease entites from OpenTargets:
{disease_entities}
"""

def curate_finetune_data(efo_id, 
                        cache_name, 
                        evidence_dir = '../PrismDB/sourceId_chembl_evidence/', 
                        curated_diseases_dir='../PrismDB/curated_diseases/'):
    
    command_str = f'''python prepare_txdata.py \
    --celltype_ppi ../PrismDB/networks/ppi_edgelists/ \
    --disease "{efo_id}" \
    --evidence_dir {evidence_dir} \
    --all_drug_targets_path ../PrismDB/{cache_name}/all_approved_oct2022.csv \
    --curated_disease_dir {curated_diseases_dir} \
    --chembl2db_path ../PrismDB/{cache_name}/chembl2db.txt \
    --disease_drug_evidence_prefix ../PrismDB/{cache_name}/disease_drug_evidence_ \
    --positive_proteins_prefix ../PrismDB/{cache_name}/positive_proteins_ \
    --negative_proteins_prefix ../PrismDB/{cache_name}/negative_proteins_ \
    --raw_data_prefix ../PrismDB/{cache_name}/raw_targets_'''
    os.system(command_str)
    
def database_update(efo_id):
    command_str = f"python retrival_protein_regulation.py \
    --task_name {efo_id} \
    --embeddings_dir ../PrismDB/pinnacle_embeds/ \
    --positive_proteins_prefix ../PrismDB/therapeutic_target_task/positive_proteins_{efo_id} \
    --negative_proteins_prefix ../PrismDB/therapeutic_target_task/negative_proteins_{efo_id}"
    os.system(command_str)
    
def search_disease_open_target(disease_name):
    # Set variables object of arguments to be passed to endpoint
    variables = {"diseaseName": disease_name}
    while True:
        try:
            # Perform POST request and check status code of response
            r = requests.post(base_url, json={"query": open_target_query_string, "variables": variables})
            # Transform API response from JSON into Python dictionary and print in console
            api_response = json.loads(r.text)
            relavent_entities = api_response['data']['search']['hits']
        except Exception as e:
            print(f"[pinnacle] Bad OpenTarget API response, retrying...")
            continue
        break
    # Check the top 5 entities
    filtered_entities = [ e for e in relavent_entities if e['entity'] == 'disease']
    return filtered_entities

def search_disease_efo(disease_name):
    """
    Search for a disease in the EFO ontology
    """
    disease_id = None
    disease_entities = search_disease_open_target(disease_name)
    query_diseaseId = QUERY_DISEASEID.format(disease_name=disease_name, disease_entities=disease_entities)
    query_diseaseId = add_message(query_diseaseId)
    while True:
        try:
            ans = chat_completion(query_diseaseId)
            disease_id = ans.split(":")[-1].strip()
        except Exception as e:
            print(f"[pinnacle] Bad request on EFO format, retrying...")
            continue
        break
    return disease_id


def regulation_dict(efo_number,
                    data = '../PrismDB/',
                    labels_file='pinnacle_embeds/pinnacle_labels_dict.txt', 
                    task_type='disease_conditioned/',
                    # embeddings_dir='pinnacle_embeds/',
                    positive_proteins_prefix='positive_proteins',
                    negative_proteins_prefix='negative_proteins'):
    
    # embed_path = data + embeddings_dir + embed + "_protein_embed.pth"
    labels_path = data + labels_file
    positive_proteins_prefix = "_".join([positive_proteins_prefix, efo_number])
    negative_proteins_prefix = "_".join([negative_proteins_prefix, efo_number])
    regulation_data_path = data + task_type
    if not os.path.exists(regulation_data_path):
        os.makedirs(regulation_data_path)
    positive_proteins_prefix = os.path.join(regulation_data_path, positive_proteins_prefix)
    negative_proteins_prefix = os.path.join(regulation_data_path, negative_proteins_prefix)
    positive_proteins, negative_proteins, _ = read_labels_from_evidence(positive_proteins_prefix, negative_proteins_prefix, None)
    assert len(positive_proteins) > 0
    # print(embed_path, labels_path)
    # print(positive_proteins_prefix, negative_proteins_prefix)
    # celltype_ppi_embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, _ = load_data(embed_path, labels_path, positive_proteins_prefix, negative_proteins_prefix, None)
    return positive_proteins, negative_proteins

def disease_conditioned_info(disease_name, data_path='../PrismDB', cache_name='disease_conditioned'):
    """
    Get information about a disease from the EFO ontology
    """
    efo_number = search_disease_efo(disease_name)
    print(f"[pinnacle] {disease_name} - {efo_number}")
    cache_path = os.path.join(data_path, cache_name)
    gene_regulation_file = [f for f in os.listdir(cache_path) if efo_number in f]
    
    if len(gene_regulation_file) == 0:
        curate_finetune_data(efo_number, cache_name)
        print(f"[pinnacle]: Curate fintuned data for {efo_number}")
        
    positive_proteins, negative_proteins = regulation_dict(efo_number, task_type=cache_name)
    
    # celltype_protein_dict - {<celltype-name>: genes-name that can celltype_ppi_embed value}
    # for k, v in celltype_protein_dict.items():
    #     print(f"[pinnacle]: {k} - {len(v)}")
    #     break
    
    # celltype_ppi_embed - {<celltype-index>: tensor(weights)}
    # for k, v in celltype_ppi_embed.items():
    #     print(f"[pinnacle]: {k} - {v.shape}")
    #     break
    
    # celltype_dict - {<celltype-name>: celltype-index}
    # for k, v in celltype_dict.items():
    #     print(f"[pinnacle]: {k} - {v}")
    #     break
    # print(len(celltype_protein_dict['acinar cell of salivary gland']))
    # print(celltype_ppi_embed[celltype_dict['acinar cell of salivary gland']].shape)
    
    print(f"[pinnacle]: Create gene regulation database under {efo_number}")
    return efo_number, positive_proteins, negative_proteins
    
if __name__ == '__main__':
    disease_conditioned_info("Neoplasm")