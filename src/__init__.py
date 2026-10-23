"""Calibration under temporal splits for CTR prediction (Avazu).

Modules:
    data_prep     -- loading, time parsing, categorical encoding, split protocols
    data_pipeline -- Plan-A streaming prepare: raw CSV -> encoded .npy splits
    models        -- Logistic Regression (SGD), DeepFM, DCN-v2
    train_eval    -- model training / prediction helpers and metric computation
    calibration   -- ECE, reliability tables, Platt / isotonic recalibration
    bidding_sim   -- second-price bidding payoff simulation
    run_experiments -- end-to-end orchestration
"""
