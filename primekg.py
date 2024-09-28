from tdc.resource import PrimeKG
# from collections import defaultdict

pkg = PrimeKG()
pkgdf = pkg.get_data()

def get_all_drug_evidence(disease):
    """given a disease, retrieve all drugs interacting with proteins relevant to disease"""
    prots = pkgdf[(pkgdf["relation"] == "disease_protein") & (pkgdf["x_name"] == disease)]["y_name"].unique()
    drugs = pkgdf[(pkgdf["relation"] == "drug_protein") & (pkgdf["y_name"].isin(prots))]
    relations = drugs["display_relation"].unique()
    out = {}
    for rel in relations:
        out[rel] = drugs[drugs["display_relation"] == rel]["x_name"].unique()
    return out

def get_all_associated_targets(disease):
    return pkgdf[(pkgdf["relation"] == "disease_protein") & (pkgdf["x_name"] == disease)][["y_name", "display_relation"]]

def get_disease_disease_associations(disease):
    return pkgdf[(pkgdf["relation"] == "disease_disease") & (pkgdf["x_name"] == disease)][["y_name", "display_relation"]]

def get_labels_from_evidence(disease):
    diseases = get_disease_disease_associations(disease)["y_name"]
    out = set()
    for d in diseases:
        targets = get_all_associated_targets(d)["y_name"].unique()
        out.update(targets)
    return list(out)

def all_diseases_by_keyword(kw):
    return pkgdf[(pkgdf["relation"] == "disease_protein") & (pkgdf["x_name"].str.contains(kw, case=False, na=False))]["x_name"].unique()

if __name__ == "__main__":
    x = all_diseases_by_keyword("autism")
    [get_all_drug_evidence(d) for d in x]
    [get_all_associated_targets(d) for d in x]
    [get_disease_disease_associations(d) for d in x]
    print([get_labels_from_evidence(d) for d in x])