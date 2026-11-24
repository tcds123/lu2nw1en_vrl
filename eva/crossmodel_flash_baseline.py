#!/usr/bin/env python
# eva/crossmodel_flash_baseline.py — deepseek-v4-flash 在 AIME24 上的跨模型 pass_rate 基线
#
# 用途: 用**同一份 skill** 在 AIME24 30 题上测 flash 的 EM pass_rate, 作为"演化后的 skill 是否
#       跨模型泛化"的对照基线 (对照组 = 同 skill 下的 DeepSeek-V3.2 历史数字)。
#
# 与 tf_global_v2_aime.main() 的关系: 复用完全相同的评测原语
#   precompute_s0 (S₀ k 投票) → run_one (eval_paired 内部逐题调用)
# main() 是"用 train 失败样本去改写 skill"的演化流程, 会做 3 轮 rw/exp/rewrite;
# 本基线**不演化**, 每个 skill 只做一次 paired 评测 (cand vs base, 同 EVAL_TEMP / 同 items → 对消共享噪声)。
#
# flash 接入 = 纯 env 替换 (qwen_backend 是通用 OpenAI 兼容 POST, 零代码改动), env 必须在
# import tf_global_v2_aime 之前设置 (tf_global_v2_math 的 setdefault 不覆盖已设值)。
#
# 用法:
#   source lunwen/bin/activate
#   nohup python eva/crossmodel_flash_baseline.py --eval_k 8 > eva/flash_baseline_20261102.log 2>&1 &
#
# 产物: <out_dir>/run_<ts>/{eval_paired_samples.jsonl, baseline_flash.json, run.log}
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_model_cfg(cfg_path: Path, provider: str):
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = (cfg.get("providers") or {}).get(provider)
    if not p:
        sys.exit(f"[err] {provider} 不在 {cfg_path}")
    return p["api_key"], p["base_url"], p.get("model", provider)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek-v4-flash")
    ap.add_argument("--config", default=str(REPO / "configs" / "models_aimp.local.yaml"))
    ap.add_argument("--data", default="/data/zhuldz/lunwen/data/aime24.jsonl")
    ap.add_argument("--s0", default="", help="默认取 tf_global_v2_aime.S0_DEFAULT (skill_aime_initial_v2.md)")
    ap.add_argument("--cand_skill", nargs="*", default=[],
                    help="额外要测的候选 skill 文件 (均为 vs S₀ paired); 缺省只测 S₀")
    ap.add_argument("--eval_k", type=int, default=8, help="每题 k 次投票 (与既有 rep5 口径一致)")
    ap.add_argument("--n_items", type=int, default=30, help="AIME24 全 30 题 (不切 split)")
    ap.add_argument("--train_n", type=int, default=18)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--s0_cache", default="", help="复用 S₀ cache json (同 model+k+temp 才可复用)")
    ap.add_argument("--out_dir", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline")
    ap.add_argument("--max_rpm", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--no_isolate_max_tokens", dest="isolate_max_tokens", action="store_false",
                    help="不隔离 (保留 run_one 内 4096 字面量), 复现历史 cap 口径时用")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"[err] 本地明文配置不存在: {cfg_path}")
    key, base_url, model = load_model_cfg(cfg_path, args.provider)

    # ---- 1. env 必须在 import evolver 之前 ----
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = base_url
    os.environ["QWEN_CHAT_MODEL"] = model
    os.environ.pop("QWEN_CHAT_AIGC_USER", None)  # openai-compat 端点不需要
    os.environ["QWEN_CHAT_MAX_RPM"] = str(args.max_rpm)
    os.environ["QWEN_CHAT_MAX_TOKENS"] = str(args.max_tokens)

    sys.path.insert(0, str(REPO / "rl" / "train" / "verl1" / "skill_evolution"))
    from tf_global_v2_aime import (  # noqa: E402
        S0_DEFAULT, EVAL_TEMP, set_temp, md5, load_aime_split,
        precompute_s0, eval_paired, bootstrap_ci_lower,
    )

    s0_path = args.s0 or S0_DEFAULT
    s0 = open(s0_path, encoding="utf-8").read().strip()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    class _Tee:
        def __init__(self, *fs):
            self.fs = fs
        def write(self, s):
            for f in self.fs:
                f.write(s); f.flush()
        def flush(self):
            import sys as _s
            for f in self.fs:
                if not getattr(f, "closed", False) and f is not _s.__stdout__:
                    try:
                        f.flush()
                    except ValueError:
                        pass

    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, logf)
    print(f"[crossmodel] provider={args.provider} model={model} base_url={base_url}")
    print(f"[crossmodel] key={key[:8]}*** (redacted)  S0={s0_path} ({len(s0)} chars, md5 {md5(s0)})")
    print(f"[crossmodel] eval_k={args.eval_k} EVAL_TEMP={EVAL_TEMP} max_rpm={args.max_rpm} max_tokens={args.max_tokens}")

    # ---- 输出长度隔离 (2026-11-02) ----
    # run_one L119 显式传 max_completion_tokens=4096 字面量, qwen_backend L282 min(4096, config.max_tokens)
    # → 即便 env 设 16384 也被钳回 4096 (F7 in-sample 遗留, docs/phase2_issues.md #18)。
    # 跨模型对比的唯一变量应是"模型", 故进程内把该字面量提为 max_tokens; 不改文件, 不影响其它 run。
    if args.isolate_max_tokens:
        import tf_global_v2_aime as _tf
        _cf = _tf.run_one.__code__
        _newc = tuple(args.max_tokens if (c == 4096) else c for c in _cf.co_consts)
        _tf.run_one.__code__ = _cf.replace(co_consts=_newc)
        n_rep = sum(1 for c in _cf.co_consts if c == 4096)
        print(f"[crossmodel] cap 隔离: run_one 内 {n_rep} 处 4096 → {args.max_tokens}")

    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    train_items, val_items = load_aime_split(args.data, args.train_n, args.split_seed)
    # n_items 只截取评测子集 (probe 用); 缺省 30 = 全量. train/val 划分仍按全量算 (train_n=18, seed=42)
    items_eval = items[:args.n_items] if args.n_items < len(items) else items
    print(f"[crossmodel] items={len(items)} (train {len(train_items)} / val {len(val_items)}), "
          f"评测子集 n={len(items_eval)}")

    # ---- 2. S₀ 基线 (k 投票, EVAL_TEMP) ----
    s0_cache = None
    if args.s0_cache and os.path.exists(args.s0_cache):
        s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
        print(f"[crossmodel] loaded s0_cache: {len(s0_cache)} 题")
    else:
        t0 = time.time()
        print(f"[crossmodel] ⏳ precompute S₀: {len(items_eval)} 题 × k={args.eval_k} = {len(items_eval)*args.eval_k} 请求 ...")
        s0_cache = precompute_s0(items_eval, s0, args.eval_k)
        print(f"[crossmodel] S₀ 完成 {(time.time()-t0)/60:.1f} min")
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache_aime.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    def pool_pass(cache, subset):
        vals = [cache.get(str(it["id"]), {}).get("pass_rate", 0.0) for it in subset]
        return sum(vals) / max(1, len(vals))

    result = {
        "ts": ts, "provider": args.provider, "model": model, "base_url": base_url,
        "eval_k": args.eval_k, "eval_temp": EVAL_TEMP, "n_items": len(items_eval),
        "s0_path": s0_path, "s0_md5": md5(s0),
        "s0_all": pool_pass(s0_cache, items_eval),
        "s0_train": pool_pass(s0_cache, train_items),
        "s0_val": pool_pass(s0_cache, val_items),
        "candidates": [],
    }
    print(f"📊 flash S₀ pass: all={result['s0_all']:.4f}  train={result['s0_train']:.4f}  val={result['s0_val']:.4f}")

    # ---- 3. 候选 skill paired 评测 (cand vs S₀, 同 EVAL_TEMP 同 items) ----
    for path in args.cand_skill:
        cand = open(path, encoding="utf-8").read().strip()
        t0 = time.time()
        print(f"\n[crossmodel] 📐 paired: {path} ({len(cand)} chars, md5 {md5(cand)}) vs S₀ ...")
        # eval_paired 返回 5 元组 (mean_pass, mean_delta, n_pairs, per_item_delta, base_per_item)
        mean_pass, d, n_pairs, per_item, base_item = eval_paired(
            items_eval, cand, s0, stage=f"flash_paired_{md5(cand)}", k=args.eval_k,
            persist_ctx={"out_dir": out_dir, "run_id": f"flash_{ts}",
                         "model_version": model, "scorer_version": "signal_beta_scr_v1",
                         "cand_hash": md5(cand), "base_hash": md5(s0)})
        rec = {"skill_path": path, "md5": md5(cand), "chars": len(cand),
               "delta": d, "mean_pass": mean_pass, "n_pairs": n_pairs,
               "elapsed_s": round(time.time() - t0, 1)}
        if per_item:
            rec["delta_ci_lower95"] = bootstrap_ci_lower(per_item, n_boot=5000)
            result["candidates"].append(rec)
            print(f"[crossmodel] delta={d:+.4f}  CI_lower(95%)={rec['delta_ci_lower95']:+.4f}  "
                  f"n_pairs={n_pairs}  ({rec['elapsed_s']}s)")
        else:
            rec["delta_ci_lower95"] = None
            result["candidates"].append(rec)
            print(f"[crossmodel] delta={d:+.4f}  (无配对项)  ({rec['elapsed_s']}s)")

    json.dump(result, open(os.path.join(out_dir, "baseline_flash.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[crossmodel] done → {out_dir}/baseline_flash.json")
    sys.stdout = sys.__stdout__
    logf.close()


if __name__ == "__main__":
    main()
