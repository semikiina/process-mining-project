import pandas as pd, pm4py
from pathlib import Path

PARENT_DIR = Path(__file__).parent
TABLES_DIR = PARENT_DIR / "Final Tables"
SAVE_PATH = PARENT_DIR / "nets"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def next_save_path(save_path: Path) -> Path:
    i = 0
    while True:
        candidate = save_path.with_name(f"{save_path.stem}_{i}{save_path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


# Currently trying to get a proper one for protocol deviation
if __name__ == "__main__":
    # Load CSVs
    event_log = pd.read_csv(TABLES_DIR / "EL_IM_PD.csv")
    cases = pd.read_csv(TABLES_DIR / "IM_PD.csv")

    # Make activity names consistent with those provided in their process model
    event_log["Activity"] = event_log["Activity"].str.replace(r"^e_custom_", "", regex=True)

    # Merge
    cases = cases.drop(columns="Name")
    event_log = event_log.merge(cases, left_on="Lead Object Id", right_on="ID", how="inner")

    # Format with case, activity, and timestamp keys
    event_log = pm4py.format_dataframe(event_log, case_id="Name", activity_key="Activity", timestamp_key="Timestamp")

    # Split into issue management and protocol deviation
    issue_management = event_log[event_log["Id"].str.startswith("Issue_")]
    protocol_deviation = event_log[event_log["Lead Object Id"].str.startswith("ProtocolDeviation_")]
    heuristics_net = pm4py.discover_heuristics_net(protocol_deviation, dependency_threshold=0.8)

    # Visualize and save
    pm4py.view_heuristics_net(heuristics_net)
    save_path = next_save_path(SAVE_PATH / "protocol_deviation.png")
    pm4py.save_vis_heuristics_net(heuristics_net, str(save_path))



    #
    # OLD CODE
    #
    # event_log = pd.read_csv(TABLES_DIR / "EL_IM_PD.csv")
    # cases = pd.read_csv(TABLES_DIR / "IM_PD.csv")
    
    # event_log["Activity"] = event_log["Activity"].str.replace(r"^e_custom_", "", regex=True)

    # event_log = event_log.merge(cases, left_on="Lead Object Id", right_on="ID", how="left")
    # event_log = pm4py.format_dataframe(event_log, case_id="ID", activity_key="Activity", timestamp_key="Timestamp")
    # # issue_management = event_log[event_log["Id"].str.startswith("Issue_")]
    # protocol_deviation = event_log[event_log["ID"].str.startswith("ProtocolDeviation_")]
    # heuristics_net = pm4py.discover_heuristics_net(protocol_deviation, dependency_threshold=0.5)

    # # Visualize and save the heuristics net
    # pm4py.view_heuristics_net(heuristics_net)
    # save_path = next_save_path(SAVE_PATH / "protocol_deviation.png")
    # pm4py.save_vis_heuristics_net(heuristics_net, str(save_path))
