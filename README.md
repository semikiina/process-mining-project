# Process Mining Analysis

A complete process mining analysis pipeline for Novo Nordisk's clinical trial data, analyzing Issues, Protocol Deviations, and Task workflows.

## Quick Start

### 1. Prepare Your Data

Create a `data/` folder in this directory and add your four CSV files:

```bash
mkdir -p data
# Copy your CSVs into data/:
# - Task.csv
# - TM.csv
# - IM_PD.csv
# - EL_IM_PD.csv
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `pm4py` - Process mining

### 3. Run the Analysis

```bash
python main.py
```

This runs all analysis steps in order:
1. **Data Analysis** - Profile and validate input CSVs
2. **Build Model** - Create object-centric event model
3. **Compute KPIs** - Calculate key performance indicators
4. **Generate Plots** - Create visualizations for the report

## Project Structure

```
PM project/
├── data/                      # Input CSV files
│   ├── Task.csv              # Task records
│   ├── TM.csv                # Task Monitoring records
│   ├── IM_PD.csv             # Issues and Protocol Deviations
│   └── EL_IM_PD.csv          # Event log
├── model/                     # Generated object-centric event model (auto-created)
│   ├── cases_issue.csv       # Issue cases with durations
│   ├── cases_pd.csv          # PD cases with durations
│   ├── events.csv            # Unified event log with links
│   ├── log_*.csv             # Flat event logs per case type
│   └── build_report.txt      # Build summary and checks
├── kpi/                       # Generated KPI reports (auto-created)
│   ├── k*.csv                # Individual KPI reports
│   ├── plots/                # Visualization outputs
│   └── kpi_report.txt        # KPI summary
├── main.py                    # Pipeline orchestrator
├── analysis.py                # Step 1: Data profiling
├── build_model.py             # Step 2: Event model construction
├── kpis.py                    # Step 3: KPI computation
├── plots.py                   # Step 4: Visualization generation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Pipeline Steps

### Step 1: Data Analysis (`analysis.py`)

Profiles and validates the input CSV files:
- Row counts, columns, null values
- Uniqueness and linking integrity
- Date ranges and computed durations
- State, category, and study distributions

**Output:** `profile_report.txt`

### Step 2: Build Model (`build_model.py`)

Constructs an object-centric event model:
- Merges the four input CSVs into a unified structure
- Creates case tables (Issue, Protocol Deviation)
- Generates event logs for each case type
- Flags and reports data quality issues

**Outputs:**
- `cases_issue.csv`, `cases_pd.csv` - Case summaries with durations
- `events.csv` - Unified event log with case links
- `log_issue.csv`, `log_pd.csv`, `log_task.csv`, `log_tm.csv` - Flat event logs
- `build_report.txt` - Row counts and validation checks

### Step 3: Compute KPIs (`kpis.py`)

Calculates key performance indicators addressing the research questions:

| KPI | Description | Use Case |
|-----|-------------|----------|
| K1 | Resolution time per case type | RQ2: How long do Issue vs PD take? |
| K2 | Long waits (≥30 days) | RQ2: Where are the bottlenecks? |
| K3 | Rework and reopenings | RQ3: How much rework is there? |
| K4 | Benchmark metrics | RQ5: How do different sites/studies compare? |
| K5 | Open and stuck cases | RQ2: What's still pending? |
| K6 | Effort baseline | Value case: Cost of rework? |

**Outputs:** Individual CSV files for each KPI + summary in `kpi_report.txt`

### Step 4: Generate Plots (`plots.py`)

Creates publication-ready visualizations:
- **fig1_resolution_time** - Cumulative resolution by day
- **fig2_long_waits** - Share of cases with 30+ day waits
- **fig3_bottlenecks** - Transitions with longest waits
- **fig4_reopenings** - Resolution time comparison
- **fig5_studies** - Study performance
- **fig6_categories** - Category breakdown
- **fig7_stuck_cases** - Open cases by inactivity duration

**Output:** PNG (200 dpi) and PDF (vector) files in `kpi/plots/`

## Data Requirements

**You must provide** four CSV files in the `data/` folder. These are NOT included in the repository (they are gitignored for size reasons).

### Expected CSV Files

- **Task.csv** - Task records with: ID, IssueObject_ID, ProtocolDeviation_ID, CreatedDate, CompletionDate, State, Category, Study, Site, etc.
- **TM.csv** - Task Monitoring follow-ups (same schema as Task.csv)
- **IM_PD.csv** - Issues and Protocol Deviations with: ID, Identifier, Name, LifecycleState, CreatedDate, ResolvedDate, Severity, Category, etc.
- **EL_IM_PD.csv** - Event log with: Id (event ID), Lead Object Id, Task Id, TaskMonitoringFollowUp Id, Timestamp, etc.

### Setup Example

```bash
# From the project root directory:
mkdir -p data
cp /path/to/your/Task.csv data/
cp /path/to/your/TM.csv data/
cp /path/to/your/IM_PD.csv data/
cp /path/to/your/EL_IM_PD.csv data/

# Verify all files are present:
ls data/
# Should output: EL_IM_PD.csv  IM_PD.csv  Task.csv  TM.csv
```

Then run `python main.py` to begin the analysis.

## Troubleshooting

### "Module not found" error
Install missing dependencies:
```bash
pip install -r requirements.txt
```

### File not found errors
Ensure all input CSV files are in the `data/` folder with correct names. Check `analysis.py` output for missing files.

### Plots not generating
Ensure matplotlib can access X11 or display. On macOS, this is usually automatic. On Linux, may need:
```bash
export MPLBACKEND=Agg
```

### Memory issues with large CSVs
The analysis loads full CSVs into memory. If you have insufficient RAM, consider:
- Processing data in chunks (modify `build_model.py`)
- Using a machine with more memory
- Filtering data before analysis (e.g., date range subsets)

## Configuration

Each script accepts command-line arguments for custom paths:

```bash
# Step 1: Profile data
python analysis.py --dir data

# Step 2: Build model
python build_model.py --dir data --ocel

# Step 3: Compute KPIs
python kpis.py --dir data

# Step 4: Generate plots
python plots.py --dir data
```

Use `python <script> --help` for full options.

## Output Interpretation

### Resolution Time (K1, fig1)
- Shows percentage of cases resolved by elapsed days
- Blue = Issue, Orange = Protocol Deviation
- Dashed line at 30 days highlights key SLA threshold

### Long Waits (K2, fig2)
- Identifies bottlenecks in the process
- Shows which transitions (state changes) consume the most time
- Waits ≥30 days considered "long"

### Rework (K3, fig4)
- Separates cases that were reopened from those resolved once
- Reopened cases take significantly longer on average

### Benchmarking (K4, fig5-fig6)
- Compares performance across sites, studies, categories, severity
- Highlights outliers and performance leaders


