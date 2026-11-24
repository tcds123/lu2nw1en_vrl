"""APPS-Plus environment adapter for SkillOpt (ReflACT).

Connects SkillOpt's train/eval loop to the APPS-Plus interview code dataset:
single-shot generation, stdin/stdout verifier (evaluator.py), ratio split over
the JSON array. No Docker, no exec backend — pure qwen_chat calls plus
subprocess-exec'd stdin/stdout comparison.
"""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.appsplus.dataloader import APPSPlusDataLoader
from skillopt.envs.appsplus.rollout import run_batch


class APPSPlusAdapter(EnvAdapter):
    """APPS-Plus adapter."""

    def build_reference_text(self, item: dict) -> str:
        return str(item.get("raw_solution") or "").strip()

    def get_reference_metadata(self, item: dict) -> dict:
        sol = str(item.get("raw_solution") or "").strip()
        if not sol:
            return {"fields": [], "preview": ""}
        return {"fields": ["raw_solution"], "preview": sol[:400]}

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 300,
        workers: int = 8,
        analyst_workers: int = 8,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
        dataset_glob: str = "data_with_id.json",
        difficulty_filter: str = "interview",
        **kwargs,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = APPSPlusDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
            dataset_glob=dataset_glob,
            difficulty_filter=difficulty_filter,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
        )

    def get_task_types(self) -> list[str]:
        return self.dataloader.get_task_types()
