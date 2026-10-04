"""Training-only pairwise target policy; listwise always retains its caller's data."""
MODES = ("coupled", "score_fixed")


def set_mode(model, mode):
    if mode not in MODES:
        raise ValueError(f"pairwise_lambda_mode must be one of {MODES}, got {mode!r}")
    model.pairwise_lambda_mode = mode
    model.save_hyperparameters({"pairwise_lambda_mode": mode})


def prepare(model, coupled_data, actual_lambda, indices, pf, ranking_pf,
            qf, score, ranking_score, prediction, bucket, material, ply):
    # Reuse the identical graph at lambda=1. No recomputation/reordering.
    if (getattr(model, "pairwise_lambda_mode", "coupled") == "coupled"
            or actual_lambda == 1.0):
        return coupled_data
    return model._prepare_sorted_data(
        indices, pf, qf, score, ranking_pf, ranking_score,
        prediction, bucket, material, ply)
