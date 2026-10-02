"""Training-only compatibility policy; no forward/export semantics."""


def validate_hysteresis_options(*, mode, qat_mode, boundary="off", factor="off",
                                local_pair="off", side=False, psqt=False,
                                bucket_importance=False):
    if mode == "off":
        return
    if qat_mode != "full":
        raise ValueError("anti_flip currently requires QAT full: weights/weight_activation "
                         "also quantize Main FT and dense weights, but this filter is "
                         "validated only with the full fixed-forward contract; "
                         "QAT off has no forward weight bins")
    if boundary != "off":
        raise ValueError("anti_flip + boundary regularization is intentionally unsupported: "
                         "overlapping regularizers have not been validated together")
    if factor != "off":
        raise ValueError("anti_flip + virtual factorization is deferred: effective-weight "
                         "history and correction distribution to specific/virtual are undefined")
    if local_pair not in ("off", "gs_local_pair32"):
        raise ValueError("anti_flip currently supports only LocalPair OFF or gs_local_pair32; "
                         "other branches are not validated")
    if side or psqt or bucket_importance:
        raise ValueError("anti_flip + side-input/PSQT/bucket-importance is not validated; "
                         "these combinations remain intentionally unsupported")
