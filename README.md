# 2025-autumn-aice

## Project Background

AI for Climate (AICE) is an interdisciplinary initiative that leverages advances in AI and expanding data availability to tackle critical challenges in climate prediction, impact modeling, and adaptation strategies. By uniting expertise from climate science, computer science, economics, physics, public health, and other fields, AICE develops novel tools—such as physics-informed predictive models and trustworthy datasets—to better understand climate dynamics and inform effective responses.

This project will transform a research benchmark for predicting the onset of the Indian monsoon into an open-source Python package. Accurate forecasting of the monsoon’s spatiotemporal onset is critical for agriculture in India, yet the performance of new AI-based weather models on this task remains largely untested. Building on a methodology developed at the University of Chicago, the package will use precipitation forecasts to estimate onset dates, validate them against station-based rainfall data, and compute key skill metrics such as onset error, miss rates, and false alarms. It will also generate a visual scorecard to enable straightforward comparisons between AI-driven and traditional numerical weather prediction models. Deliverables include a PyPI package that implements the full benchmark using tools such as Xarray and Dask, providing researchers and stakeholders with a reproducible framework for evaluating monsoon prediction skill.


Rapid advances in artificial intelligence (AI) have enabled AI models to potentially outperform traditional numerical weather prediction (NWP) at weather prediction. However, most AI forecast evaluation studies have compared models using global metrics over limited years without focusing on sector and region specific applications. Operationally driven benchmarking is necessary to effectively deploy these models, informing both model selection and improvements for different decision-making needs. In this work, we introduce a human-centered benchmarking framework that is operational, scientific, and farmer-centric and use it to assess the performance of several state-of-the-art AI models in forecasting local-scale agriculturally relevant monsoon onset over India. Model performance is evaluated against rain gauge observations using both deterministic and probabilistic metrics. Our work presents a framework for developing operational human-centered benchmarks, which can accelerate the translation of AI-driven advances to weather forecasting applications worldwide.

## First Week
- Complete the quick start below, making sure that you can find the file `sample_output.csv`.
- Read this introduction to [Packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- Read the paper uploaded to the group [Box folder](https://uchicago.app.box.com/folder/343188541726)
- Download the data from the Box folder to your local machine and place it in a directory you can point to in the `.env` file – see the instructions below.
- Examine the scripts and corresponding README files in the `reference_scripts` directory to understand the current benchmarking workflow. (reference output files can be found in the Box folder: `/monsoon-benchmark/reference_scripts/*/output/`)  

### Data download instructions 
Copy the following directories into your `data` directory that you had specified during the setup of the Clinic environment:  
- `model_forecast_data`: Contains model forecast data files. 
- `imd_rainfall_data`: Contains IMD rainfall NetCDF files. 
- `imd_onset_threshold`: Contains monsoon onset threshold NetCDF files. 

This will take up approximately 21GB of disk space. If you are running low on disk space, it is not necessary to download `imd_rainfall_data/0p25/` nor `imd_rainfall_data/1p0/` as we are only using the 2.0 and 4.0 degree data at the present.  

## Goal: Develop a python package that takes in precipitation forecasts and ground truth data to score the model performance in predicting the spatio-temporal onset of the Indian Monsoon


## Data:
- Gridded, station-based daily rainfall measurements over India (1901–2024) [NetCDF]
- 35-Day precipitation forecasts over India from multiple weather models (May-July, 1965–2024) [NetCDF]


## Ground-truth and Forecast Onset Calculations
- Be able to use default threshold values (as directed in Moron & Robinson 2014) or specified threshold values 
- Change the number of days and amount of rain for the wetspell criterion
- Forecast benchmark metric computation 
- Be able to handle sparse forecast initiations (e.g. handle forecasts initiated daily in addition to those initiated twice weekly)
- Be able to choose which forecast initiations to evaluate (e.g. all, only Mondays & only Thursdays, etc.)
- Save benchmark results in a serialized format, alongside metadata 
- Be able to change the probabilistic onset threshold from 50% of members to a specified percentage
- Data visualization in the form of a model scorecard comparison and onset map 
- Export high quality visualizations in a specified format, with png as the default

### Additional Specifications
- Must follow FAIR data principles 
- Include all relevant metadata alongside results in a serialized format
- Transition from command-line scripts to object-oriented programming API (WeatherBenchX is a great example)
- If there is any redundant code shared among the scripts, generalize it into a function that can be used by all functions or classes
- Consider the use of classes when constructing an evaluation schema
- If any intermediate files do not already exist, the code must be able to generate and cache them as needed (e.g. any threshold files)
- Save a config alongside all benchmarks that are run, including all specifications to allow for full reproducibility, which can include but are not limited to:
  - Years validated 
  - Model evaluated 
  - Ground-truth dataset used
  - Thresholds used 
  - Forecast length evaluated (e.g. 1-15 or 16-30 days)
  - Initiation dates
  - Use of MOK filter or not
- Must be able to handle both deterministic and probabilistic forecasts 
- Implement proper error handling (e.g. when a deterministic forecast is used as input into a probabilistic metric, when ground-truth and forecast resolutions differ)
- Visualizations must be thoughtfully designed and easily interpretable 
- Bonus if visualizations are made to be both interactive in a web-based dashboard and static in summary output plots (can be an option for the following quarter or if time permits)

### Output Metrics
- Mean Absolute Error (MAE)
- False Alarm Rate (FAR)
- Miss Rate (MR)
- Brier Score (BS)
- Ranked Probability Score (RPS)
- Area Under the Receiver Operating Characteristic Curve (AUC)
- Reliability Diagram 

#### Bonus Output Metrics
- Probability of detection 
- Critical success index

For more detailed information on these metrics, including their formulation, see the supporting material in the box folder. 

### Core Python Packages
- [Xarray](https://docs.xarray.dev/en/stable/) – allows for reading multi-dimensional datasets, think Pandas but supercharged. Please review this [crash course](https://tutorial.xarray.dev/overview/xarray-in-45-min.html) on Xarray if you are unfamiliar. 
- [Dask](https://docs.dask.org/en/stable/) – allows for lazy loading and parallelization with Xarray, improving memory efficiency. See this [article](https://docs.xarray.dev/en/latest/user-guide/dask.html) for more information about Dask-based parallelization. 
- [CDO](https://code.mpimet.mpg.de/projects/cdo) – primarily used for regridding NetCDF files in this project. Students will likely not need to use it, but it must be included for end users. 
- [NetCDF4](https://unidata.github.io/netcdf4-python/)\* 
- [H5NetCDF](https://h5netcdf.org/index.html)\* 

\* required for interacting with NetCDF files via Python and must be installed for Xarray to work properly

### Project-Specific Use of Generative AI Policy

This project recognizes the value of AI-based tools, but also their limitations. Generative AI should be treated as a supporting tool, not a replacement for human work. All code and documentation in this project must be written by a human to ensure full understanding of its functionality. AI may be used as a teaching aid–for example, to clarify unfamiliar Python packages or data types–but never as a shortcut to avoid learning. Its role is to support and enhance understanding, not diminish it.


## Quick Start

### 1. Setup Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your data directory path
# Example: DATA_DIR=/Users/yourname/project/data
```

### 2. Install Pre-commit Hooks
```bash
make run-interactive
# Inside container:
cd src
pre-commit install
exit
```

### 3. Test Your Setup
```bash
make test-pipeline
```

If successful, you should see `sample_output.csv` appear in your data directory.

## Technical Expectations

### Pre requisites:

We use Docker, Make and uv as part of our curriculum. If you are unfamiliar with them, it is strongly recommended you read over the following:
- [An introduction to Docker](https://docker-curriculum.com/)
- [An introduction to uv](https://realpython.com/python-uv/)

### Container-Based Development

**All code must be run inside the Docker container.** This ensures consistent environments across different machines and eliminates "works on my machine" issues.

### Environment Management with uv

We use [uv](https://docs.astral.sh/uv/) for Python environment and package management _inside the container_. uv handles:
- Virtual environment creation and management (replaces venv/pyenv)
- Package installation and dependency resolution (replaces pip)
- Project dependency management via `pyproject.toml`

**Important**: When running Python code, prefix commands with `uv run` to maintain the proper environment:

```bash
# Example: Running the pipeline
uv run python src/utils/pipeline_example.py

# Example: Running a notebook
uv run jupyter lab

# Example: Running tests
uv run pytest
```

### Container Volume Structure

```
Container: /project/
├── src/           # Your source code (mounted from host repo)
├── data/          # Data directory (mounted from HOST_DATA_DIR)
├── .venv/         # Python virtual environment (created in container)
├── pyproject.toml # Project configuration
└── ...
```


## Usage & Testing

- Set `DATA_DIR` in your `.env` file to specify where data lives on your host
- This directory is mounted to `/project/data` inside the container
- Keep data separate from code to avoid repository bloat and enable easy data sharing

Run the command `make test-pipeline`. If your setup is working you should see a file `sample_output.csv` appear in your data directory. 


### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a Jupyter server, which also mounts the current directory into `/program`.
* `make run-interactive`: This will create a container (with the current directory mounted as `/program`) and load an interactive session. 

The file `Makefile` contains details about the specific commands that are run when calling each `make` target.




## Style
We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

Once you have followed the quick setup instructions above for installing dependencies, you can run:
```bash
pre-commit run --all-files
```

You can also run `ruff` directly:
```bash
ruff check
ruff format
```

### Visualization Data Downloads

Visualization utilities now expose a standardized way to export the gridded
metrics used for scorecards and onset maps. After generating a
``spatial_metrics`` dictionary, call:

```python
from monsoonbench.visualization import download_spatial_metrics_data

download_spatial_metrics_data(
    spatial_metrics,
    output_dir="artifacts",
    formats=("netcdf", "csv"),
    metadata={"model": "GraphCast"},
)
```

This writes well-documented files that can be fed into downstream plotting
experiments or shared with collaborators.

From the CLI you can emit the exact same bundle while generating plots/NetCDF
results:

```bash
monsoonbench \
  --config configs/deterministic.yaml \
  --download_dir artifacts \
  --download_formats netcdf csv parquet
```

Optional flags such as `--download_metrics mean_mae miss_rate` and
`--download_keep_nans` let you tailor the export for downstream consumers.