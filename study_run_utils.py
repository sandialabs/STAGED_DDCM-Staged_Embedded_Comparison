from __future__ import annotations

from pathlib import Path
import json
import pickle


STAGE_SPECS = {
    "embedded": {
        "summary": "summary.json",
        "error": "error.json",
        "required_outputs": [
            "observations.npz",
            "split.npz",
            "embedded_model.eqx",
            "embedded_model_config.json",
            "embedded_closure.npz",
            "summary.json",
        ],
        "prerequisites": [],
    },
    "enkf": {
        "summary": "enkf_summary.json",
        "error": "enkf_error.json",
        "required_outputs": [
            "enkf_result.npz",
            "enkf_summary.json",
        ],
        "prerequisites": [
            "observations.npz",
            "split.npz",
        ],
    },
    "staged": {
        "summary": "staged_summary.json",
        "error": "staged_error.json",
        "required_outputs": [
            "staged_model.eqx",
            "staged_model_config.json",
            "staged_closure.npz",
            "staged_summary.json",
        ],
        "prerequisites": [
            "observations.npz",
            "split.npz",
            "enkf_result.npz",
        ],
    },
}


def rep_dir(study_dir, r: int) -> Path:
    return Path(study_dir) / f"rep_{r:03d}"


def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_pickle_if_exists(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def has_files(rep_path: Path, filenames: list[str]) -> bool:
    return all((rep_path / name).exists() for name in filenames)


def stage_is_successful(rep_path: Path, stage: str) -> bool:
    """
    A stage is considered complete if:
        1. all required output files exist;
        2. the stage summary JSON exists;
        3. summary["success"] is True.

    If a summary file does not contain "success", this returns False.
    """
    spec = STAGE_SPECS[stage]

    if not has_files(rep_path, spec["required_outputs"]):
        return False

    summary = load_json_if_exists(rep_path / spec["summary"])

    if summary is None:
        return False

    return bool(summary.get("success", False))


def stage_has_failed(rep_path: Path, stage: str) -> bool:
    """
    A stage is considered failed if its stage-specific error file exists,
    or if its summary exists with success=False.
    """
    spec = STAGE_SPECS[stage]

    if (rep_path / spec["error"]).exists():
        return True

    summary = load_json_if_exists(rep_path / spec["summary"])

    if summary is not None and summary.get("success", True) is False:
        return True

    return False


def stage_prerequisites_exist(rep_path: Path, stage: str) -> bool:
    spec = STAGE_SPECS[stage]

    return has_files(rep_path, spec["prerequisites"])


def select_replicates_to_run(
    study_dir,
    num_replicates: int,
    stage: str,
    *,
    overwrite: bool = False,
    rerun_failed: bool = True,
    require_prerequisites: bool = True,
):
    """
    Select replicate IDs that should be run for a given stage.

    Parameters
    ----------
    study_dir:
        Top-level study directory.
    num_replicates:
        Desired total number of replicate IDs, interpreted as range(num_replicates).
    stage:
        One of "embedded", "enkf", or "staged".
    overwrite:
        If True, run all requested replicates regardless of existing outputs.
    rerun_failed:
        If True, failed/incomplete replicates are included.
        If False, failed replicates are skipped.
    require_prerequisites:
        If True, skip replicates that do not have prerequisite files for this stage.

    Returns
    -------
    tasks:
        List of replicate IDs to submit.
    skipped_success:
        List of replicate IDs skipped because stage already succeeded.
    skipped_failed:
        List of replicate IDs skipped because stage failed and rerun_failed=False.
    skipped_missing_prereq:
        List of replicate IDs skipped because prerequisites are missing.
    """
    if stage not in STAGE_SPECS:
        raise ValueError(
            f"Unknown stage '{stage}'. Expected one of {sorted(STAGE_SPECS)}."
        )

    study_dir = Path(study_dir)

    tasks = []
    skipped_success = []
    skipped_failed = []
    skipped_missing_prereq = []

    for r in range(num_replicates):
        rp = rep_dir(study_dir, r)

        if overwrite:
            if require_prerequisites and not stage_prerequisites_exist(rp, stage):
                skipped_missing_prereq.append(r)
            else:
                tasks.append(r)
            continue

        if stage_is_successful(rp, stage):
            skipped_success.append(r)
            continue

        if stage_has_failed(rp, stage) and not rerun_failed:
            skipped_failed.append(r)
            continue

        if require_prerequisites and not stage_prerequisites_exist(rp, stage):
            skipped_missing_prereq.append(r)
            continue

        tasks.append(r)

    return tasks, skipped_success, skipped_failed, skipped_missing_prereq


def print_replicate_selection_summary(
    *,
    stage: str,
    num_replicates: int,
    tasks: list[int],
    skipped_success: list[int],
    skipped_failed: list[int],
    skipped_missing_prereq: list[int],
):
    print("\n" + "=" * 80)
    print(f"Stage: {stage}")
    print("=" * 80)
    print(f"Requested replicate count:          {num_replicates}")
    print(f"Replicates selected to run:         {len(tasks)}")
    print(f"Replicates skipped, successful:     {len(skipped_success)}")
    print(f"Replicates skipped, failed:         {len(skipped_failed)}")
    print(f"Replicates skipped, missing prereq: {len(skipped_missing_prereq)}")

    if len(tasks) > 0:
        print(f"First selected replicates: {tasks[:10]}")
        print(f"Last selected replicates:  {tasks[-10:]}")

    print("=" * 80 + "\n")