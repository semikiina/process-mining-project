from pathlib import Path
import pandas as pd, pm4py
from pm4py.algo.evaluation.generalization import algorithm as gen_eval

PARENT_DIR = Path(__file__).parent
TABLES_DIR = PARENT_DIR / "data"
SAVE_PATH = PARENT_DIR / "nets"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def next_save_path(save_path: Path) -> Path:
    i = 0
    while True:
        candidate = save_path.with_name(f"{save_path.stem}_{i}{save_path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def evaluate(log, name, model):
    net, im, fm = model
    fit = pm4py.fitness_token_based_replay(log, net, im, fm)
    sound = pm4py.analysis.check_is_sound(net, im, fm)
    return {
        "model": name,
        "fitness": fit["average_trace_fitness"],
        "pct_fitting": fit["perc_fit_traces"],
        "precision": pm4py.precision_token_based_replay(log, net, im, fm),
        "generalization": gen_eval.apply(log, net, im, fm),
        "sound": sound[0] if isinstance(sound, tuple) else sound,
    }


# Currently just doing protocol deviation
if __name__ == "__main__":
    # Load CSVs
    event_log = pd.read_csv(TABLES_DIR / "EL_IM_PD.csv", low_memory=False)
    cases = pd.read_csv(TABLES_DIR / "IM_PD.csv", low_memory=False)

    # Filter out unused columns
    event_log = event_log[["Id", "Name", "Activity", "Timestamp", "Lead Object Id"]]
    cases = cases[["ID"]]

    # Make activity names consistent with those provided in their process model
    event_log["Activity"] = event_log["Activity"].str.replace(r"^e_custom_", "", regex=True)

    # Merge
    event_log = event_log.merge(cases, left_on="Lead Object Id", right_on="ID", how="inner")

    # Format with case, activity, and timestamp keys
    event_log = pm4py.format_dataframe(event_log, case_id="Name", activity_key="Activity", timestamp_key="Timestamp")

    # Split into issue management and protocol deviation
    issue_management = event_log[event_log["Lead Object Id"].str.startswith("Issue_")]
    protocol_deviation = event_log[event_log["Lead Object Id"].str.startswith("ProtocolDeviation_")]

    # Discover heuristics net
    # heuristics_net = pm4py.discover_heuristics_net(protocol_deviation, dependency_threshold=0.8)

    # Visualize and save
    # pm4py.view_heuristics_net(heuristics_net)
    # save_path = next_save_path(SAVE_PATH / "protocol_deviation.png")
    # pm4py.save_vis_heuristics_net(heuristics_net, str(save_path))

    dependency_threshold = 0.8
    models = {f"HM_{dependency_threshold}": pm4py.discover_petri_net_heuristics(protocol_deviation, dependency_threshold=dependency_threshold)}
    noise_threshold = 0.0
    models[f"IM_{noise_threshold}"] = pm4py.discover_petri_net_inductive(protocol_deviation, noise_threshold=noise_threshold)

    results = pd.DataFrame([evaluate(protocol_deviation, n, m) for n, m in models.items()]).set_index("model")
    print(results.round(3))
