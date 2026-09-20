"""Lightweight immutable run registry and frozen nominal-policy loading."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from dataclasses import asdict
import yaml

from credit_rl.config import SimulationConfig
from credit_rl.risk.longitudinal import LongitudinalPDModel
from .compare_policies import load_specs, run_id
from .common import write_manifest


def write_json(path, data):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def load_frozen(seeds=None):
    base=SimulationConfig.from_yaml("configs/simulation.yaml")
    benchmark=yaml.safe_load(Path("configs/policy_evaluation.yaml").read_text())
    risk=LongitudinalPDModel.load(benchmark["pd_model"])
    specs=load_specs(base,benchmark,Path("outputs/results/policy_evaluation"),Path("outputs/models/policy_evaluation"),run_id(base,benchmark))
    if seeds is not None:
        specs=[s for s in specs if s.name not in ("PPO","PPO_without_PD") or s.seed in seeds]
    return base,benchmark,risk,specs


def register(experiment, profile, settings, base, benchmark, specs, extra):
    folder=Path("outputs/experiments")/f"{experiment}_{profile}"; folder.mkdir(parents=True,exist_ok=True)
    core=list(Path("src/credit_rl/simulation").glob("*.py"))+list(Path("src/credit_rl/policies").glob("*.py"))
    core += [Path(p) for p in ("src/credit_rl/config.py","src/credit_rl/reward.py",
        "src/credit_rl/envs/credit_limit_env.py","src/credit_rl/envs/observation.py",
        "src/credit_rl/envs/constraints.py","src/credit_rl/evaluation/policy_engine.py",
        "src/credit_rl/evaluation/worlds.py","src/credit_rl/evaluation/sensors.py")]
    if experiment=="ope":
        core += [Path("src/credit_rl/evaluation/ope.py"),Path("experiments/off_policy_evaluation.py")]
    else:
        core += [Path("experiments/robustness.py")]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(core)}
    artifact_hashes={s.model_identifier:hashlib.sha256(Path(s.model_identifier).read_bytes()).hexdigest()
        for s in specs if s.model_identifier != "rule"}
    identity_payload=dict(settings=settings,profile=profile,simulation=asdict(base),benchmark=benchmark,
        core_sources=hashes,policies=artifact_hashes,pd_sha256=hashlib.sha256(Path(benchmark["pd_model"]).read_bytes()).hexdigest(),extra=extra)
    identity=hashlib.sha256(json.dumps(identity_payload,sort_keys=True).encode()).hexdigest()
    path=folder/"manifest.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["experiment_id"]!=identity:
            raise ValueError(f"Inputs changed for {folder}; preserve/move the completed run or use a different profile")
        # Also protect risk-feature/preprocessing sources captured by the full manifest.
        # Report-only modules may evolve without invalidating expensive simulated outcomes.
        for name,digest in old["source_sha256"].items():
            source=Path(name)
            if source.parts[:3] == ("src","credit_rl","risk"):
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f"Risk pipeline source changed: {source}")
    else:
        commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
        write_manifest(path,base,settings,experiment=experiment,experiment_id=identity,
            timestamp=datetime.now(timezone.utc).isoformat(),git_commit=commit,**identity_payload)
        for name in ("simulation","macro_scenarios","pd_model","policy_evaluation",experiment):
            source=Path("configs")/(name+".yaml")
            if source.exists(): (folder/source.name).write_bytes(source.read_bytes())
    write_json(folder/"policy_inventory.json",[dict(policy=s.name,policy_seed=s.seed,
        information_set=s.information_set,model_identifier=s.model_identifier,without_pd=s.without_pd) for s in specs])
    return identity,folder
