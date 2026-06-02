<h1 align="center">
	<!-- Replace the image URL with your project logo when available -->
	<img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/python.svg" width="120" alt="data-analysis-tool logo">
	<br>
	data-analysis-tool
</h1>

<p align="center">
	A practical Python toolkit for data inspection, preprocessing, and exploratory plots.
</p>

[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/Sasindu99-ai/data-analysis)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-not%20published-lightgrey.svg)](https://pypi.org/)

- **Website:** https://github.com/Sasindu99-ai/data-analysis
- **Documentation:** https://github.com/Sasindu99-ai/data-analysis
- **Source code:** https://github.com/Sasindu99-ai/data-analysis
- **Bug reports:** https://github.com/Sasindu99-ai/data-analysis/issues

It centers around two main entry points: `DataInspector` for dataset cleanup and
summary, and `PlottingMethods` for quick visual analysis with Plotly.

## Features

- Dataset summary: shapes, column types, missing values, and duplicates.
- Preprocessing pipeline with reusable transformers (type correction, missing values,
  outliers, duplicates, and column/row removal).
- Numeric and categorical feature encoding with merge utilities.
- Plotly-based exploratory charts (univariate, relationships, categorical counts,
  and association heatmaps).

## Installation

This project is not published on PyPI yet. Install from source:

```bash
python -m pip install .
```

For development tools:

```bash
python -m pip install .[dev]
```

## Quick start

### Inspect a dataset

```python
import pandas as pd

from data_analysis import DataInspector, Workspace

df = pd.read_csv("data.csv")

inspector = DataInspector(workspace=Workspace.LOCAL)
inspector.load_dataframe(df)
inspector.summary()
```

### Run a preprocessing pipeline

```python
from data_analysis import (
	DataInspector,
	Workspace,
	AutoTypeCorrector,
	MissingValueHandler,
	DuplicateRemover,
	ColumnRemover,
)

inspector = DataInspector(workspace=Workspace.LOCAL)
inspector.load_dataframe(df)

inspector.pipeline([
	("auto_type", AutoTypeCorrector()),
	("missing_values", MissingValueHandler()),
	("duplicates", DuplicateRemover()),
	("drop_columns", ColumnRemover(["id"]))
])

clean_df = inspector.preprocess(show_summary=True)
```

### Normalize and merge features

```python
from data_analysis import (
	DataInspector,
	Workspace,
	NumericNormalizeMethod,
	CategoricalNormalizeMethod,
)

inspector = DataInspector(workspace=Workspace.LOCAL)
inspector.load_dataframe(df)

num_df = inspector.extract_normalized_numeric_data(
	method=NumericNormalizeMethod.MINMAX
)
cat_df = inspector.extract_normalized_categorical_data(
	method=CategoricalNormalizeMethod.ONEHOT
)

merged_df = inspector.merge_normalized_data()
```

### Plot common analyses

```python
from data_analysis import PlottingMethods

plots = PlottingMethods(df)

univariate_figs = plots.plot_numeric_univariate(["age", "income"])
relation_fig = plots.plot_relationship("age", "income")
freq_fig = plots.plot_categorical_frequency("country")
heatmap_fig = plots.plot_all_associations_heatmap()
```

## API highlights

- `DataInspector`: load data, summarize, and run preprocessing pipelines.
- `PlottingMethods`: exploratory plots backed by Plotly figures or HTML snippets.
- Pipeline transformers: `AutoTypeCorrector`, `ColumnRemover`, `DuplicateRemover`,
  `MissingValueHandler`, `MissingValueSanitizer`, `OutlierHandler`, `RowRemover`.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, scipy, plotly

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome. Please add tests for any new features and
run formatters before submitting a PR.

