from tdc.benchmark_group import scdti_group
from tdc.resource.pinnacle import PINNACLE
from pandas import DataFrame
import pandas as pd

def get_dti_dataset():
    group = scdti_group.SCDTIGroup()
    train_val = group.get_train_valid_split()
    train = train_val["train"]
    val = train_val["val"]
    test = group.get_test()["test"]
    return pd.concat([train, val, test], axis=0, ignore_index=True)

def get_embedddings():
    pinnacle = PINNACLE()
    embeds = pinnacle.get_embeds()
    assert isinstance(embeds, DataFrame)
    assert len(embeds) > 0, "PINNACLE embeds is empty"
    keys = pinnacle.get_keys()
    assert isinstance(keys, DataFrame)
    assert len(keys) > 0, "PINNACLE keys is empty"
    assert len(keys) == len(embeds), "{} vs {}".format(
        len(keys), len(embeds))
    num_targets = len(keys["target"].unique())
    num_cells = len(keys["cell type"].unique())
    all_entries = embeds.index
    prots = [x.split("--")[0] for x in all_entries] # proteins are entry 0
    cells = [x.split("--")[1] for x in all_entries] # cells are entry 1
    embeds["protein"] = prots
    embeds["cell"] = cells
    return embeds

def get_ctspec_protein_embed(cell, protein):
    df = get_embedddings()
    result = df[(df['cell'] == cell) & (df['protein'] == protein)]
    return result

def is_target(cell, protein, disease):
    df = get_dti_dataset()
    result = df[(df['cell'] == cell) & (df['name'] == protein) & (df["disease"] == disease)]
    return result["y"] == 1

def get_ppi():
    return PINNACLE().get_ppi()

def get_cell_mg():
    return PINNACLE().get_mg()


