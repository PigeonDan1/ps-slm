def ctc_greedy_decode(pred_ids, blank_id=0):
    """
    Collapse repeats and remove blank tokens.
    pred_ids: 1D tensor or list
    """
    if hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()
    new_ids = []
    prev = None
    for i in pred_ids:
        if i != prev and i != blank_id:
            new_ids.append(i)
        prev = i
    return new_ids
