import torch


def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db


def emr_db_tuning(logits, targets):
    dbs = torch.linspace(0, 1, 100)
    num_examples = targets.shape[0]
    exact_matches = torch.zeros((len(dbs)))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        exact_matches[idx] = torch.all(torch.eq(predictions, targets), dim=-1).sum()

    best_emr = exact_matches.max() / num_examples
    best_db = dbs[exact_matches.argmax()]
    print(f"Best EMR: {best_emr} at DB: {best_db}")
    return best_emr, best_db
