# Experiment results and evaluation notes

## Historical results

The attached report and the saved Colab outputs refer to different model runs. The report lists MAE 228.519, RMSE 351.445 and R² 0.9686 without enough checkpoint information to match it to the saved notebook. The main README therefore uses the traceable final LSTM evaluation from source cell 49:

- MAE: 226.8134002685547
- RMSE: 347.03415267809015
- R²: 0.969403088092804
- Parameters: hidden size 64, 2 layers, dropout 0.2, Adam learning rate 0.001, batch size 64, Huber loss.
- Best validation Huber loss: 0.01517796459861871; checkpoint at epoch 18.

Cell references are zero-based in the original source notebook, before the explanatory header added to the archived copy.

The first comparison's lowest test RMSE (340.487208) belongs to LSTM + Adam with no additional L1/L2 penalty. It is not reported as the final unbiased score: configurations were ranked by test RMSE. The final LSTM grid search selected by validation, but used the same test set already inspected in the exploratory comparison.

The generic `model` variable left after the comparison loop holds GRU + Adam + L2, not the best LSTM. Thus source cells 30/31 and 52–54 concern that GRU. The recorded recursive RMSE 663.72097 must not be attributed to the selected LSTM. Recursive evaluation also keeps measured weather covariates, so it is conditional rather than a fully ex-ante multi-step forecast.

## Corrections in the runnable notebook

- Labels refer to records/observations, not unverified hourly or daily lags.
- Identical timestamps stay in one chronological partition. Cut points use approximately 70/15/15 percent of unique timestamps; row proportions may differ.
- Weather-category encoding is fit on training data only and handles unseen categories.
- StandardScaler is fit on training data only, as in the original notebook.
- The optimizer comparison reports validation RMSE without test access.
- Hyperparameter search and checkpoint selection use validation loss.
- Final test evaluation occurs after selection; outputs from the old execution are not attached to modified code.
- Recursive evaluation uses `best_run['model']` explicitly.
- Persistence baselines invert the traffic feature's own scaler statistics.
- Random seeds and saved preprocessing/checkpoints support repeatable follow-up runs.

## Remaining limitations

1. This is a next-record experiment. Duplicate timestamp records and missing hours prevent a fixed one-hour forecast interpretation. Duplicate target values can make next-record prediction easier.
2. For a real hourly benchmark, first consolidate repeated timestamps, define a regular hourly grid and use only contiguous windows with observed targets; specify how missing weather and traffic are handled without future information.
3. Reusing the historical test interval cannot make it unseen again. A stronger follow-up uses a fresh holdout or a predeclared rolling-origin protocol.
4. The rainfall filter `rain_1h < 40` is inherited. Temperature zero and other possible data-quality issues need a documented treatment in a subsequent benchmark.
5. Ordinal weather encoding imposes an artificial order; one-hot encoding is a useful comparison.
6. No uncertainty estimates, repeated-seed study, deployment latency or production monitoring were measured.
7. No fresh training was performed during packaging. The current environment lacks PyTorch and the source CSV. Syntax and preprocessing/window invariants were checked on synthetic data only.

## Figures

The PNGs are extracted from original Colab outputs, avoiding report-page screenshots and preserving plot quality. `historical_lstm_forecast.png` is the selected LSTM (source cell 50). `historical_gru_forecast.png` is the final model from the initial comparison (source cell 31).
