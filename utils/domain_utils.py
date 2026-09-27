def get_domain_score_key(domain):
    if domain == "imo_proof":
        return "points_percentage"
    elif domain == "bbob_unconstrained":
        return "score"
    elif domain == "bbob_constrained":
        return "score"
    elif domain == "metabox_mo":
        return "score"


def get_domain_splits(domain, eval_test=False):
    if domain in ["search_arena", "paper_review", "imo_grading"]:
        splits = ["train", "val"]
        if eval_test:
            splits.append("test")
        return splits
    elif domain == "bbob_unconstrained":
        return ["train"]
    elif domain == "bbob_constrained":
        return ["train"]
    elif domain == "metabox_mo":
        return ["train"]


def can_domain_ensembled(domain):
    if domain in ["search_arena", "paper_review", "imo_grading"]:
        return True
    elif domain == "imo_proof":
        return False
    elif domain == "bbob_unconstrained":
        return False
    elif domain == "bbob_constrained":
        return False
    elif domain == "metabox_mo":
        return False


def get_domain_eval_subset(domain):
    if domain in ["search_arena", "paper_review"]:
        return "_filtered_100_train"
    elif domain == "imo_grading":
        return "_filtered_100_train"
    elif domain == "bbob_unconstrained":
        return ""
    elif domain == "bbob_constrained":
        return ""
    elif domain == "metabox_mo":
        return ""


def get_domain_test_subset(domain):
    if domain in ["search_arena", "paper_review"]:
        return "_filtered_100_test"
    elif domain == "imo_grading":
        return "_filtered_100_test"
    elif domain == "bbob_unconstrained":
        return ""
    elif domain == "bbob_constrained":
        return ""
    elif domain == "metabox_mo":
        return ""


def get_domain_stagedeval_samples(domain):
    if domain == "bbob_constrained":
        return -1
    elif domain == "bbob_unconstrained":
        return -1
    elif domain == "metabox_mo":
        return -1


def get_domain_stagedeval_frac(domain):
    if domain == "bbob_constrained":
        return 54/54
    elif domain == "bbob_unconstrained":
        return 24/24
    elif domain == "metabox_mo":
        return 35/35


def has_domain_val_subset(domain):
    if domain in ["search_arena", "paper_review", "imo_grading"]:
        return True
    elif domain == "bbob_unconstrained":
        return False
    elif domain == "bbob_constrained":
        return False
    elif domain == "metabox_mo":
        return False