# BÁO CÁO TỔNG HỢP TOÀN DIỆN CÁC PHƯƠNG PHÁP DUNG HỢP ENSEMBLE
**Tập dữ liệu:** Expanded 54 Đặc trưng SOTA (Chunked Test: 372,993 dòng)

## Bảng Xếp Hạng Toàn Bộ Các Phương Pháp Dung Hợp (Sorted by PR-AUC)

| Method / Ensemble Strategy | Type | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Best Threshold (tau*) | TP | FP | FN | TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Trio_Prob (XGB + CAT + LGBM) | Soft Voting | 0.845 | 0.9522 | 0.8398 | 0.9312 | 0.7648 | 0.2473 | 6982 | 516 | 2149 | 363346 |
| Quad_Harmonic_Mean | Harmonic Mean | 0.8448 | 0.9528 | 0.842 | 0.9273 | 0.771 | 0.1809 | 7039 | 552 | 2092 | 363310 |
| **Quad_Geometric_Mean** | **Geometric Mean** | 0.8447 | 0.953 | 0.8437 | 0.9405 | 0.765 | 0.2342 | 6983 | 442 | 2148 | 363420 |
| Trio_Rank (XGB + CAT + LGBM) | Rank Averaging | 0.8446 | 0.9522 | 0.8403 | 0.9226 | 0.7715 | 0.979 | 7042 | 590 | 2089 | 363272 |
| Quad_Weighted_Rank (PR-AUC) | Rank Weighted | 0.8443 | 0.9528 | 0.8428 | 0.9477 | 0.7587 | 0.9797 | 6928 | 386 | 2203 | 363476 |
| Quad_SLSQP_Rank_Averaging | Rank SLSQP | 0.8443 | 0.9528 | 0.8428 | 0.9479 | 0.7587 | 0.9797 | 6928 | 385 | 2203 | 363477 |
| **Quad_Equal_Rank_0.25 (CHAMPION)** | Rank Averaging | 0.8443 | 0.9528 | 0.8428 | 0.9479 | 0.7587 | 0.9797 | 6928 | 384 | 2203 | 363478 |
| Pair_Prob (XGB + CAT) | Soft Voting | 0.8441 | 0.9507 | 0.843 | 0.9362 | 0.7666 | 0.2296 | 6999 | 477 | 2132 | 363385 |
| Pair_Rank (CAT + LGBM) | Rank Averaging | 0.8441 | 0.9513 | 0.8448 | 0.945 | 0.7639 | 0.9799 | 6963 | 405 | 2168 | 363457 |
| Trio_Rank (CAT + LGBM + HIST) | Rank Averaging | 0.8436 | 0.9524 | 0.8448 | 0.9514 | 0.7597 | 0.9799 | 6933 | 350 | 2198 | 363512 |
| Quad_Equal_Prob (Soft Voting) | Soft Voting | 0.8436 | 0.9531 | 0.844 | 0.9461 | 0.7617 | 0.2818 | 6955 | 396 | 2176 | 363466 |
| Quad_Weighted_Prob (PR-AUC) | Soft Voting Weighted | 0.8436 | 0.9531 | 0.8439 | 0.9462 | 0.7615 | 0.2824 | 6952 | 395 | 2179 | 363467 |
| Pair_Prob (XGB + LGBM) | Soft Voting | 0.8434 | 0.9534 | 0.8331 | 0.9142 | 0.7652 | 0.2113 | 6986 | 656 | 2145 | 363206 |
| Pair_Rank (XGB + CAT) | Rank Averaging | 0.8434 | 0.9506 | 0.8419 | 0.9355 | 0.7654 | 0.9794 | 6991 | 485 | 2140 | 363377 |
| Trio_Rank (XGB + CAT + HIST) | Rank Averaging | 0.8434 | 0.952 | 0.8443 | 0.9473 | 0.7616 | 0.9795 | 6954 | 392 | 2177 | 363470 |
| Pair_Rank (XGB + LGBM) | Rank Averaging | 0.8432 | 0.9534 | 0.8316 | 0.9178 | 0.7603 | 0.9795 | 6938 | 619 | 2193 | 363243 |
| Trio_Rank (XGB + LGBM + HIST) | Rank Averaging | 0.8431 | 0.9537 | 0.8356 | 0.93 | 0.7586 | 0.9795 | 6923 | 521 | 2208 | 363341 |
| Quad_Min_Probability (Conservative) | Min Voting | 0.8429 | 0.9515 | 0.8393 | 0.9532 | 0.7498 | 0.1584 | 6846 | 336 | 2285 | 363526 |
| Single_LightGBM | Single Model | 0.8423 | 0.9537 | 0.8333 | 0.9171 | 0.7634 | 0.2502 | 6971 | 630 | 2160 | 363232 |
| Pair_Rank (LGBM + HIST) | Rank Averaging | 0.842 | 0.9538 | 0.8382 | 0.9397 | 0.7565 | 0.9799 | 6908 | 443 | 2223 | 363419 |
| Single_XGBoost | Single Model | 0.8406 | 0.9521 | 0.8326 | 0.9422 | 0.7459 | 0.2814 | 6810 | 418 | 2321 | 363444 |
| Quad_Max_Probability (Optimistic) | Max Voting | 0.8397 | 0.9527 | 0.8357 | 0.9418 | 0.7512 | 0.4868 | 6859 | 424 | 2272 | 363438 |
| Single_CatBoost | Single Model | 0.8388 | 0.9459 | 0.8433 | 0.9488 | 0.759 | 0.2593 | 6930 | 374 | 2201 | 363488 |
| Single_HistGradientBoosting | Single Model | 0.834 | 0.9522 | 0.8268 | 0.9262 | 0.7467 | 0.3341 | 6818 | 543 | 2313 | 363319 |

## Trọng Số Tối Ưu SLSQP (Non-linear Constrained Optimization)
- Trọng số tối ưu hóa trên Validation: `[XGB=0.250, CAT=0.250, LGB=0.250, HIST=0.250]`
- **Kết luận thực nghiệm:** SLSQP hội tụ tuyệt đối về trọng số đều $[0.25, 0.25, 0.25, 0.25]$, tương đương 100% với **Equal Rank Averaging 0.25**.
