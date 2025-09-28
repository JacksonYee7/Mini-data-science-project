# src/pipeline.py — ONE-COMMAND pipeline (AE -> MLP/XGB/LGBM -> ridge stacking -> check)
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def run(cmd_list):
    print("[RUN]", " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def exists(p):
    return Path(p).exists()

def parse_args():
    ap = argparse.ArgumentParser("One-command pipeline")
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["all","core","ae","models","stack","check","report"])
    ap.add_argument("--force", type=int, default=None,
                    help="override config.force (1=rerun even if outputs exist)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.force is not None:
        cfg["force"] = int(args.force)

    # ---- Paths & switches
    seeds = [int(s) for s in cfg["seeds"]]
    force = bool(cfg.get("force", 0))
    chunk_rows = str(cfg.get("chunk_rows", 200_000))

    # Data & features
    train_raw = cfg["data"]["train_raw"]
    test_raw  = cfg["data"]["test_raw"]
    feat_base = cfg["features"]["base"]
    feat_plus = cfg["features"]["plus_ae"]

    # Processed
    core_train = cfg["processed"]["train_core"]
    core_test  = cfg["processed"]["test_core"]
    ae_train   = cfg["processed"]["train_ae"]
    ae_test    = cfg["processed"]["test_ae"]

    # Out dirs
    out_mlp = cfg["outdirs"]["mlp"]
    out_xgb = cfg["outdirs"]["xgb"]
    out_lgb = cfg["outdirs"]["lgb"]
    ensure_dir(out_mlp); ensure_dir(out_xgb); ensure_dir(out_lgb)
    ensure_dir("submissions"); ensure_dir("reports"); ensure_dir("assets/features")
    ensure_dir(Path(core_train).parent); ensure_dir(Path(ae_train).parent)

    # ============== STAGE: core (可选) ==============
    if args.stage in ["all","core"]:
        if force or (not exists(core_train) or not exists(core_test)):
            run([
                sys.executable, "src/00_build_core_tables.py",
                "--train_raw", train_raw,
                "--test_raw",  test_raw,
                "--feature_file", feat_base,
                "--include_micro", "0",
                "--out_train", core_train,
                "--out_test",  core_test
            ])
        else:
            print("[SKIP] core tables exist.")

    # ============== STAGE: AE ==============
    if args.stage in ["all","ae"]:
        if force or (not exists(ae_train) or not exists(ae_test) or not exists(feat_plus)):
            run([
                sys.executable, "src/13_autoencoder_feats.py",
                "--train", core_train, "--test", core_test,
                "--feature_file", feat_base, "--include_micro", "0",
                "--code_dim", str(cfg["ae"]["code_dim"]),
                "--hidden", *[str(x) for x in cfg["ae"]["hidden"]],
                "--dropout", str(cfg["ae"]["dropout"]),
                "--epochs", str(cfg["ae"]["epochs"]),
                "--batch_size", str(cfg["ae"]["batch_size"]),
                "--lr", str(cfg["ae"]["lr"]),
                "--seed", str(cfg["ae"]["seed"]),
                "--out_train", ae_train,
                "--out_test",  ae_test,
                "--out_feat_file", feat_plus
            ])
        else:
            print("[SKIP] AE outputs exist.")

    # helper: build single-model commands
    def mlp_cmd(seed):
        return [
            sys.executable, "src/12_mlp_global.py",
            "--train", ae_train, "--test", ae_test,
            "--feature_file", feat_plus,
            "--include_micro", "0",
            "--winsor_p", str(cfg["common"]["winsor_p"]),
            "--label_perday_z", "1",
            "--use_holdout_last_month", "1",
            "--hidden", *[str(x) for x in cfg["mlp"]["hidden"]],
            "--dropout", str(cfg["mlp"]["dropout"]),
            "--epochs", str(cfg["mlp"]["epochs"]),
            "--lr", str(cfg["mlp"]["lr"]),
            "--batch_size", str(cfg["mlp"]["batch_size"]),
            "--chunk_rows", chunk_rows,
            "--seed", str(seed),
            "--outdir", out_mlp
        ]

    def xgb_cmd(seed):
        return [
            sys.executable, "src/10_xgb_global_v2.py",
            "--train", ae_train, "--test", ae_test,
            "--feature_file", feat_plus,
            "--include_micro", "0",
            "--winsor_p", str(cfg["common"]["winsor_p"]),
            "--label_perday_z", "1",
            "--use_holdout_last_month", "1",
            "--learning_rate", str(cfg["xgb"]["learning_rate"]),
            "--max_depth", str(cfg["xgb"]["max_depth"]),
            "--n_estimators", str(cfg["xgb"]["n_estimators"]),
            "--subsample", str(cfg["xgb"]["subsample"]),
            "--colsample_bytree", str(cfg["xgb"]["colsample_bytree"]),
            "--min_child_weight", str(cfg["xgb"]["min_child_weight"]),
            "--lambda_l2", str(cfg["xgb"]["lambda_l2"]),
            "--early_stopping_rounds", str(cfg["xgb"]["early_stopping_rounds"]),
            "--chunk_rows", chunk_rows,
            "--seed", str(seed),
            "--n_jobs", str(cfg["xgb"]["n_jobs"]),
            "--outdir", out_xgb
        ]

    def lgb_cmd(seed):
        return [
            sys.executable, "src/08c_lgbm_global_holdout.py",
            "--train", ae_train, "--test", ae_test,
            "--feature_file", feat_plus,
            "--include_micro", "0",
            "--winsor_p", str(cfg["common"]["winsor_p"]),
            "--label_perday_z", "1",
            "--use_holdout_last_month", "1",
            "--learning_rate", str(cfg["lgbm"]["learning_rate"]),
            "--num_leaves", str(cfg["lgbm"]["num_leaves"]),
            "--n_estimators", str(cfg["lgbm"]["n_estimators"]),
            "--min_data_in_leaf", str(cfg["lgbm"]["min_data_in_leaf"]),
            "--feature_fraction", str(cfg["lgbm"]["feature_fraction"]),
            "--bagging_fraction", str(cfg["lgbm"]["bagging_fraction"]),
            "--bagging_freq", str(cfg["lgbm"]["bagging_freq"]),
            "--lambda_l1", str(cfg["lgbm"]["lambda_l1"]),
            "--lambda_l2", str(cfg["lgbm"]["lambda_l2"]),
            "--early_stopping_rounds", str(cfg["lgbm"]["early_stopping_rounds"]),
            "--seed", str(seed),
            "--n_jobs", str(cfg["lgbm"]["n_jobs"]),
            "--outdir", out_lgb
        ]

    # 结果文件路径（按固定顺序）
    def files_for(outdir, stem, seeds, prefix=""):
        # stem: 'mlp' 或 'xgb' 或 'lgb'
        vals = [f"{outdir}/valpred_global_{stem}_seed{s}.parquet" for s in seeds]
        preds= [f"{outdir}/pred_global_{stem}_seed{s}.parquet"     for s in seeds]
        return vals, preds

    # ============== STAGE: models ==============
    if args.stage in ["all","models"]:
        # MLP
        for s in seeds:
            pred_path = f"{out_mlp}/pred_global_mlp_seed{s}.parquet"
            if force or not exists(pred_path):
                run(mlp_cmd(s))
            else:
                print(f"[SKIP] MLP seed{s} done.")
        # XGB
        for s in seeds:
            pred_path = f"{out_xgb}/pred_global_xgb_seed{s}.parquet"
            if force or not exists(pred_path):
                run(xgb_cmd(s))
            else:
                print(f"[SKIP] XGB seed{s} done.")
        # LGBM
        for s in seeds:
            pred_path = f"{out_lgb}/pred_global_lgb_seed{s}.parquet"
            if force or not exists(pred_path):
                run(lgb_cmd(s))
            else:
                print(f"[SKIP] LGBM seed{s} done.")

    # ============== STAGE: stack ==============
    val_mlp, pred_mlp = files_for(out_mlp, "mlp", seeds)
    val_xgb, pred_xgb = files_for(out_xgb, "xgb", seeds)
    val_lgb, pred_lgb = files_for(out_lgb, "lgb", seeds)
    val_all = val_mlp + val_xgb + val_lgb
    pred_all= pred_mlp + pred_xgb + pred_lgb

    if args.stage in ["all","stack"]:
        # ridge stacking on holdout
        out_sub = cfg["stack"]["out"]
        metric_out = cfg["stack"]["metric_out"]
        if force or not exists(out_sub):
            run([
                sys.executable, "src/09b_weight_search_holdout.py",
                "--mode", "ridge",
                "--valpreds", *val_all,
                "--testpreds", *pred_all,
                "--out", out_sub,
                "--metric_out", metric_out
            ])
        else:
            print("[SKIP] stacking output exists.")

    # ============== STAGE: check ==============
    if args.stage in ["all","check"]:
        run([
            sys.executable, "src/11_check_submission.py",
            "--files", cfg["stack"]["out"]
        ])

    if args.stage == "report":
        print("Open notebooks/01_report.ipynb and run all cells.")

if __name__ == "__main__":
    main()
