"""Select original validation batches containing held-out expert studies."""


def expert_replay_batches(validation_ids, expert_ids):
    validation_ids, expert_ids = list(validation_ids), list(expert_ids)
    if (len(set(validation_ids)) != len(validation_ids)
            or len(set(expert_ids)) != len(expert_ids)
            or not expert_ids or not set(expert_ids).issubset(validation_ids)):
        raise ValueError("invalid held-out expert/validation identities")
    expert = set(expert_ids)
    batches = []
    for start in range(0, len(validation_ids), 2):
        ids = validation_ids[start:start + 2]
        if expert.intersection(ids):
            batches.append((start, ids))
    return batches
