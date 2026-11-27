#!/usr/bin/env python3
"""RQ3-B: 用目标模型**自己的**失败池跑现有 Evolver (Arm1), 算法与协议不变。

不改 `tf_skill_optimizer_appsplus.py` (生产代码), 只把三处 S₀ 来源换成目标模型自己的:
  --s0_cache_train  outputs/rq3/<model>/train_s0_cache.json   (RQ3-A 产出, 80 题)
  --s0_cache_val    outputs/rq3/<model>/val_s0_cache.json     (scripts/rq3_b_eval_val.py, 100 题)
  --s0_cache_test   outputs/rq3/<model>/test_s0_cache.json    (全量 2765 test 题, issue #122)
  --best_results    ⚠️ 仍指向 val cache: test 段 "vs SkillOpt best" 对照因此无效 (id 零交集),
                    该段输出一律不用; 不影响 independent/routed/retention 三段主结论。

依据 (docs/phase2_issues_202611.md #119): 生产 launcher 复用 flash S₀ cache 会把 flash 能力混进
Arm1 的门控与 routed 分, 故 RQ3 在目标模型上重算 S₀。算法 (算子/门控/iters/g/池上限) 完全沿用默认。
"""
from __future__ import annotations

import json
import os
import sys

import yaml

ROOT = "/data/zhuldz/lunwen"
SKILLOPT = "/data/zhuldz/skillopt"
SE = os.path.join(ROOT, "rl/train/verl1/skill_evolution")
YAML = os.path.join(ROOT, "configs/models_aimp.local.yaml")


def load_test_ids():
    """test split 的全部 id (2765 题), 用于校验 test S₀ cache 覆盖度。"""
    p = os.path.join(SKILLOPT, "data/appsplus_capped_seed42/test/items.json")
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    lst = obj if isinstance(obj, list) else list(obj.values())
    return lst


def main() -> int:
    model = os.environ.get("RQ3_MODEL", "deepseek-v4-pro")
    out_dir = os.environ.get("RQ3_OUT", os.path.join(ROOT, "outputs/rq3", model, "arm1"))
    train_cache = os.path.join(ROOT, "outputs/rq3", model, "train_s0_cache.json")
    val_cache = os.path.join(ROOT, "outputs/rq3", model, "val_s0_cache.json")
    for p in (train_cache, val_cache):
        if not os.path.exists(p):
            print(f"[rq3b] BLOCKED: 缺 {p}", file=sys.stderr)
            return 2

    providers = yaml.safe_load(open(YAML, encoding="utf-8"))["providers"]
    if model not in providers:
        print(f"[rq3b] BLOCKED: yaml 无 provider {model}", file=sys.stderr)
        return 2

    # 与 scripts/rq3_a_s0_train.py / launch_tf_appsplus_routing.sh 完全一致的 backend 参数
    # (端点默认 = pro 的 OpenAI 兼容端点; GLM-5.2 用 RQ3_BASE_URL 覆盖到 aimp-glm/v1, 仅换路由)
    os.environ["QWEN_CHAT_API_KEY"] = providers[model]["api_key"]
    os.environ["QWEN_CHAT_BASE_URL"] = os.environ.get("RQ3_BASE_URL",
                                                      "https://aimpapi.midea.com/t-aigc/llm-openai-api")
    os.environ["QWEN_CHAT_MODEL"] = model
    if os.environ.get("RQ3_AIGC_USER"):
        os.environ["QWEN_CHAT_AIGC_USER"] = os.environ["RQ3_AIGC_USER"]
    else:
        os.environ.pop("QWEN_CHAT_AIGC_USER", None)
    os.environ["QWEN_CHAT_MAX_TOKENS"] = "16384"
    os.environ["QWEN_CHAT_TEMPERATURE"] = "0.0"
    os.environ["QWEN_CHAT_ENABLE_THINKING"] = "false"
    os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = "600"
    os.environ["QWEN_CHAT_MAX_RPM"] = os.environ.get("QWEN_CHAT_MAX_RPM", "20")
    os.environ["QWEN_CHAT_BURST"] = os.environ.get("QWEN_CHAT_BURST", "2")
    os.environ["TARGET_BACKEND"] = "qwen_chat"
    os.environ["OPTIMIZER_BACKEND"] = "qwen_chat"
    sys.path.insert(0, SKILLOPT)
    sys.path.insert(0, SE)

    os.makedirs(out_dir, exist_ok=True)

    # 生产代码要求 --s0*_cache_val/test/best_results 为 **list**(逐题 dict), 而 precompute_s0
    # 风格产物是 {id: dict}。这里写一份 list 形态副本 (字段不变), 供 val/test 侧读取。
    def _as_list(cache_path, list_path):
        with open(cache_path, encoding="utf-8") as f:
            obj = json.load(f)
        lst = obj if isinstance(obj, list) else list(obj.values())
        with open(list_path, "w", encoding="utf-8") as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
        return lst

    val_list_path = os.path.join(out_dir, "val_s0_cache_list.json")
    _as_list(val_cache, val_list_path)

    # test 侧 S₀ cache 必须是目标模型在**全量 test split** 上的逐题结果 (issue #122):
    # 早期版本误用 val_list_path → test id 与 val id 零交集 → eval_independent 的 McNemar /
    # base_hard / routed 诊断全部退化为空, 只有 test_independent_results.json 有效。
    test_cache = os.environ.get("RQ3_TEST_CACHE",
                                os.path.join(ROOT, "outputs/rq3", model, "test_s0_cache.json"))
    if not os.path.exists(test_cache):
        print(f"[rq3b] BLOCKED: test S₀ cache 不存在: {test_cache}\n"
              f"       先跑 scripts/rq3_b_eval_test_s0.py 生成全量 test S₀ 再启动训练",
              flush=True)
        return 3
    test_list = _as_list(test_cache, os.path.join(out_dir, "test_s0_cache_list.json"))
    test_ids = {str(r["id"]) for r in test_list}
    test_split_ids = {str(it["id"]) for it in load_test_ids()}
    missing = test_split_ids - test_ids
    if missing:
        print(f"[rq3b] BLOCKED: test S₀ cache 缺 {len(missing)} 道 test 题 (示例 {sorted(missing)[:5]})",
              flush=True)
        return 3
    print(f"[rq3b] test S₀ cache: {len(test_list)} 题 (覆盖 test split {len(test_split_ids)} 题)",
          flush=True)

    argv = [
        "tf_skill_optimizer_appsplus.py",
        "--s0", os.path.join(SKILLOPT, "skillopt/envs/appsplus/skills/initial.md"),
        "--split_dir", os.path.join(SKILLOPT, "data/appsplus_capped_seed42"),
        "--s0_cache_val", val_list_path,
        "--s0_cache_test", os.path.join(out_dir, "test_s0_cache_list.json"),  # issue #122
        "--s0_cache_train", train_cache,
        "--best_results", val_list_path,    # 仅用于解析 id; 不与 flash 211 比较 (见 issue #122#4)
        "--iters", os.environ.get("RQ3_ITERS", "3"),
        "--train_n", os.environ.get("RQ3_TRAIN_N", "80"),
        "--g", os.environ.get("RQ3_G", "4"),
        "--num_exp", "3",
        "--max_pool", "12",
        "--ablation_arm", "full",
        "--out_dir", out_dir,
    ]
    print(f"[rq3b] model={model} → {out_dir}", flush=True)
    print(f"[rq3b] train={train_cache}\n[rq3b] val ={val_cache}", flush=True)
    print(f"[rq3b] argv={' '.join(argv[1:])}", flush=True)

    import runpy
    sys.argv = argv
    runpy.run_path(os.path.join(SE, argv[0]), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
