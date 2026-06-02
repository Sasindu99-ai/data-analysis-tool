import traceback
from abc import abstractmethod, ABC
from enum import Enum
from pathlib import Path
from tkinter import Tk
from typing import Any, Self

import numpy as np
import pandas as pd
from IPython.display import display
from pandas import DataFrame
from pygments.lexers import go
from scipy.stats import chi2_contingency
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler
)
from plotly import express as px

__all__ = [
    "DataInspector", "NumericNormalizeMethod", "CategoricalNormalizeMethod", "Workspace", "Scaler", "AutoTypeCorrector",
    "ColumnRemover", "DuplicateRemover", "MissingValueHandler", "MissingValueSanitizer", "OutlierHandler", "RowRemover",
    "StandardizeFormator"
]


class Scaler(ABC):
    """
    Class Scaler(ABC)

    Description: Abstract base class for scalers. This class defines the interface for scaling data,
    including methods for fitting the scaler to the data, transforming the data, and a combined
    fit-transform method. Subclasses of Scaler should implement the abstract methods to provide
    specific scaling techniques, such as standardization, normalization, or min-max scaling.

    Methods:
        fit: Abstract method for fitting the scaler to the data.
        transform: Abstract method for transforming the data using the fitted scaler.
        fit_transform: Combines the fit and transform methods to perform both operations in one step.
        verbose: Abstract method for providing detailed information about the scaler's operation.
    """

    @abstractmethod
    def fit(self, X, y = None) -> Self:
        """
        Abstract method for fitting the scaler to the data. This method should be implemented
        in subclasses to compute and store relevant statistics or transformations necessary
        to scale the input data. The exact behavior may vary depending on the specific
        scaling technique used in the subclass.

        :param X: Input data to be fit by the scaler. It is typically structured as a 2D
            array where rows represent samples and columns represent features.
        :type X: array-like
        :param y: Target values for supervised scaling, optional. Not all scalers utilize
            the target values, and some may completely ignore this parameter.
        :type y: array-like, optional
        :return: Returns an instance of the scaler after fitting it to the provided data.
            Generally, this is the same instance on which the method was called, allowing
            for method chaining.
        :rtype: Scaler
        :raises NotImplementedError: If the method is called directly on the base class.
        """
        return self

    @abstractmethod
    def transform(self, X):
        """
        Represents an abstract transformation that operates on input data and produces an
        output. This class serves as a blueprint for all transformation subclasses that
        should implement the `transform` method. The transformation may include processing
        tasks such as data normalization, filtering, or encoding.

        Methods:
            transform: Abstract method intended to perform transformation on the input data
            and return the transformed output.

        :param X: Input data to be transformed. The structure and type of the input depend
            on the implementation in the subclass.
        :return: The transformed data. Specific type and structure depend on the
            subclass's implementation of the transformation logic.
        """
        pass

    def fit_transform(self, X, y = None):
        """
        Fits the model to the data and then transforms the data using the fitted model.

        This method first fits the model on the provided data using the `fit` method
        and then applies the transformation to the data using the `transform` method.

        :param X: Input data to fit and transform.
        :param y: Target values (default is None).
        :return: Transformed data after fitting the model.
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def verbose(self):
        pass


class Workspace(Enum):
    LOCAL = 1
    COLAB = 2


class NumericNormalizeMethod(Enum):
    MINMAX = ("minmax", MinMaxScaler)
    STD = ("std", StandardScaler)
    ROBUST = ("robust", RobustScaler)


class CategoricalNormalizeMethod(Enum):
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    UNIFORM = "uniform"


class DataInspector:
    """
    class DataInspector

    Description:
        Represents a utility class for inspecting and analyzing datasets.

        Provides methods to upload, load, and evaluate datasets from various formats such
        as CSV, Excel, and JSON. Includes functionalities to determine the environment
        (local or Google Colab), summarize the dataset structure, data types, missing
        values, and other essential details for further data processing.

    Attributes:
        workspace (Workspace): Represents the workspace environment for current
            operations. Can indicate whether the code is running locally or in
            Google Colab.
        file (Path): The path to the dataset file uploaded or selected for analysis.
            This attribute specifies the file processed by the instance.
        df (pd.DataFrame): The main pandas DataFrame object used for dataset storage
            and analysis. All loaded data is represented in this attribute.
        normalized_numeric_df (pd.DataFrame): Contains normalized numerical
            columns derived from the primary DataFrame (df). Helps in standardizing
            numerical data for uniform processing.
        normalized_categorical_df (pd.DataFrame): Contains preprocessed
            categorical data from the primary DataFrame (df). This attribute
            provides a consolidated form of encoded categorical columns.
        merged_normalized_df (pd.DataFrame): Combines the normalized numerical
            and categorical DataFrames into a comprehensive form for further
            analysis or modeling.

    Methods:
        is_colab: Determines if the current execution environment is Google Colab.
        upload_data: Prompts the user to upload a data file or specifies a file path
        load_dataframe: Loads a DataFrame into the instance, either from the provided
        df_ready: Determines if the dataframe (df) has been loaded and is non-empty.
        summary: Provides a summary of the dataset with information about its structure, contents, and missing data.
        pipeline: Configures a sequence of processing steps as a pipeline.
        preprocess: Preprocesses the data using a pipeline of processing steps, with options for verbosity, error handling, and displaying a summary.
        extract_normalized_numeric_data: Extract and normalize numeric data columns from a DataFrame based on the specified normalization method.
        extract_normalized_categorical_data: Extracts and normalizes categorical data from the dataframe based on the specified normalization method.
        merge_normalized_data: Combines the normalized numerical and categorical DataFrames into a comprehensive form for further analysis or modeling.
        plot_missing_values: Visualizes missing values in the dataset using Matplotlib.
    """

    workspace: Workspace = NotImplemented
    file: Path = NotImplemented
    df: pd.DataFrame = NotImplemented
    _pipeline: list[tuple[str, Scaler]] = []
    normalized_numeric_df: pd.DataFrame = NotImplemented
    normalized_categorical_df: pd.DataFrame = NotImplemented
    merged_normalized_df: pd.DataFrame = NotImplemented

    def __init__(self, workspace: Workspace) -> None:
        """
        Initializes an instance of the class with a given workspace.

        :param workspace: The workspace used to initialize the class instance. It acts as the main
            environment within which operations related to this instance are managed.
        :type workspace: Workspace
        """
        self.workspace = workspace

    def is_colab(self) -> bool:
        """
        Determines whether the current execution environment is Google Colab.

        This method checks if the workspace is explicitly set as `Workspace.COLAB`. If the
        workspace is not predefined (`NotImplemented`), it prompts the user to confirm
        whether they are running in Google Colab.

        :return: True if the environment is identified as Google Colab, False otherwise.
        :rtype: bool
        """
        return (
            input("Are you running this on Colab? (y/n): ").lower() == "y"
            if self.workspace is NotImplemented else self.workspace == Workspace.COLAB
        )

    def upload_data(self, file_path: str | Path | None = None) -> None:
        """
        Prompts the user to upload a data file or specifies a file path directly. This method
        supports both local file selection and Google Colab file upload functionality.

        :param file_path: Optional path to the data file. If not specified, the method prompts
                          the user to select a file manually or uploads a file in a Colab
                          environment.
        :type file_path: str | Path | None
        :return: None
        """
        if file_path is None:
            is_colab = self.is_colab()
            if is_colab:
                try:
                    from google.colab import files

                    uploaded = files.upload()
                    self.file = Path(next(iter(uploaded)))
                except ImportError:
                    print("Google Colab not detected. Please upload a file manually.")
            else:
                from tkinter.filedialog import askopenfilename
                root = Tk()
                root.withdraw()
                self.file = Path(askopenfilename(
                    title="Select a Data File",
                    filetypes=[("Data Files", "*.csv *.xlsx *.json")]
                ))
        else:
            self.file = Path(file_path)

    def load_dataframe(self, dataframes: pd.DataFrame | None = None) -> None:
        """
        Loads a DataFrame into the instance, either from the provided parameter
        or from a previously uploaded file. If no DataFrame is provided and no
        file has been uploaded, raises a ValueError. Handles files with the
        formats CSV, Excel, and JSON, raising a ValueError if the file format
        is not supported.

        :param dataframes: An optional pandas DataFrame to directly set for the instance.
                           If 'NotImplemented', attempts to load a DataFrame from
                           a provided file.
        :type dataframes: pd.DataFrame or type(NotImplemented)

        :raises ValueError: If no DataFrame is provided and no file has been uploaded.
        :raises ValueError: If the uploaded file is of an unsupported format.
        """
        if dataframes is not None:
            self.df = dataframes
            return
        if self.file is NotImplemented:
            raise ValueError("Data file has not been uploaded yet.")
        if self.file.suffix == ".csv":
            self.df = pd.read_csv(self.file)
        elif self.file.suffix == ".xlsx":
            self.df = pd.read_excel(self.file)
        elif self.file.suffix == ".json":
            self.df = pd.read_json(self.file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV, Excel, or JSON file.")

    def df_ready(self) -> bool:
        """
        Determines if the dataframe (df) has been loaded and is non-empty. Provides a
        validation step to ensure the dataframe is in a state ready for further operations.
        Raises an exception if the dataframe is uninitialized.

        This method additionally prints a message and returns False if the dataframe
        is empty but valid.

        :return: True if the dataframe is loaded and non-empty, False if the dataframe
                 is empty. Raises ValueError if the dataframe has not been implemented.
        :rtype: bool
        """
        if self.df is NotImplemented:
            raise ValueError("Dataframe has not been loaded yet.")
        if self.df.empty:
            print("The DataFrame is empty. No missing values to sanitize.")
            return False
        return True

    def summary(self, verbose: bool = True) -> dict[str, dict[str, int] | list[Any] | DataFrame | Any] | None:
        """
        Provides a summary of the dataset with information about its structure,
        contents, and missing data. It includes details such as the number of rows
        and columns, lists of numerical and categorical columns, duplicate row count,
        and statistics about missing values.

        :param verbose: Whether to display the summary interactively or return it as a
            structured dictionary.
        :type verbose: bool
        :return: A dictionary containing information about the dataset, including its
            shape, numerical and categorical columns, duplicate row count, column
            information, and missing values data.
        :rtype: dict[str, dict[str, int] | list[Any] | DataFrame | Any] | None
        :raises ValueError: If the dataframe has not been loaded yet.
        """
        if not self.df_ready():
            return None

        rows, cols = self.df.shape
        numerical_columns = (
            self.df.select_dtypes(include=["number"])
            .columns
            .tolist()
        )
        categorical_columns = (
            self.df.select_dtypes(exclude=["number"])
            .columns
            .tolist()
        )
        duplicate_count = self.df.duplicated().sum()
        info_df = pd.DataFrame({
            "Column": self.df.columns,
            "Data Type": self.df.dtypes.values,
            "Non-Null Count": self.df.notnull().sum().values,
            "Null Count": self.df.isnull().sum().values
        })
        missing_df = pd.DataFrame({
            "Column": self.df.columns,
            "Missing Values": self.df.isnull().sum().values,
            "Missing %": (
                    (self.df.isnull().sum() / len(self.df)) * 100
            ).round(2).values
        })

        if verbose:
            print("=" * 70)
            print("DATASET SUMMARY")
            print("=" * 70)

            print(f"Rows    : {rows}")
            print(f"Columns : {cols}")

            print(f"Numerical Columns   : {len(numerical_columns)}")
            print(f"Categorical Columns : {len(categorical_columns)}")

            print(f"Duplicate Rows      : {duplicate_count}")

            print("=" * 70)

            print("\nFIRST 20 ROWS")
            display(self.df.head(20))

            print("\nCOLUMN INFORMATION")
            display(info_df)

            print("\nMISSING VALUES")
            display(missing_df)

            print("\nNUMERICAL COLUMNS")
            print(numerical_columns)

            print("\nCATEGORICAL COLUMNS")
            print(categorical_columns)

            print("=" * 70)
            print("SUMMARY COMPLETED")
            print("=" * 70)

        return {
            "shape": {
                "rows": rows,
                "columns": cols
            },
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "duplicate_rows": duplicate_count,
            "column_info": info_df,
            "missing_values": missing_df
        }

    def pipeline(self, steps: list[tuple[str, Scaler]]) -> None:
        """
        Configures a sequence of processing steps as a pipeline. Each step is
        defined by a tuple containing a name and a scaler instance. The pipeline
        is stored as a list of these tuples.

        :param steps: List of tuples where each tuple consists of a string (step
            name) and a Scaler instance representing the processing step.
        :type steps: list[tuple[str, Scaler]]
        :return: None
        """
        self._pipeline = list(steps)

    def preprocess(
            self, y = None, verbose: bool = True, show_summary: bool = True, stop_on_error: bool = True
    ) -> pd.DataFrame | None:
        """
        Preprocesses the data using a pipeline of processing steps, with options for verbosity,
        error handling, and displaying a summary.

        This method applies a sequence of processing steps to the input data, updating the
        internal dataframe (`self.df`) in place. Each step in the pipeline is expected to
        contain a name and a processor, where the processor can implement `fit_transform`
        or separate `fit` and `transform` methods.

        :param y: Target variable values, optional. Used if the processors require this
                  additional input for fitting or transforming.
        :type y: pandas.Series, optional
        :param verbose: Indicates whether to display detailed progress information
                        during preprocessing.
        :type verbose: bool
        :param show_summary: Indicates whether to display a summary of the processed
                             dataframe at the end of the pipeline.
        :type show_summary: bool
        :param stop_on_error: If True, stops the pipeline execution upon encountering
                              an error in any processing step. If False, continues
                              to the next step.
        :type stop_on_error: bool
        :return: The processed dataframe if preprocessing succeeds, otherwise None.
        :rtype: pandas.DataFrame or None
        :raises ValueError: If an invalid step format is encountered within the pipeline
                            and `stop_on_error` is True.
        """

        if not self.df_ready():
            return None

        if verbose:
            print("=" * 70)
            print("STARTING PREPROCESSING PIPELINE")
            print("=" * 70)

        total_steps = len(self._pipeline)

        for index, step in enumerate(self._pipeline):
            if len(step) != 2:
                msg = f"Invalid step format at index {index}. Each step should be a tuple of (name, processor)."
                if not stop_on_error:
                    raise ValueError(msg)
                print(msg)
                continue

            name, processor = step

            try:
                if verbose:
                    print(f"\n[{index + 1}/{total_steps}] Running: {name}")
                if hasattr(processor, "fit_transform"):
                    X = processor.fit_transform(self.df)
                else:
                    processor.fit(self.df, y)
                    X = processor.transform(self.df)

                if X is None:
                    print(f"Processor {name} returned None. Skipping.")
                    continue

                if verbose:
                    processor.verbose()

                self.df = X

                if verbose:
                    print(f"✓ {name} completed")
            except Exception as e:
                print(
                    f"\n✗ Error in step {name}"
                )
                print(f"Reason: {str(e)}")

                if verbose:
                    traceback.print_exc()

                if stop_on_error:
                    print(
                        "\nPipeline stopped."
                    )
                    return None

        if show_summary:
            self.summary()

        return self.df

    def extract_normalized_numeric_data(
            self, method: NumericNormalizeMethod = NumericNormalizeMethod.MINMAX, columns: list[str] | None = None,
            verbose: bool = True
    ) -> pd.DataFrame | None:
        """
        Extract and normalize numeric data columns from a DataFrame based on the specified normalization method.

        This method processes numeric columns in the DataFrame, applies the given normalization method,
        and stores the resulting normalized data as an additional attribute. If requested, the method
        also prints verbose output detailing the operation.

        :param method: The normalization method to apply. Must be an instance of NumericNormalizeMethod.
        :param columns: A list of column names to normalize. If None, all numeric columns in the DataFrame
            are selected.
        :param verbose: A flag to enable or disable verbose output. If True, detailed progress and summary
            of the operation are printed.
        :return: A pandas DataFrame containing the normalized numeric data, or None if no numeric columns
            are available or if the input DataFrame is not valid.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        columns = [col for col in columns if col in df.columns]

        if not len(columns):
            print("No numeric columns found in the DataFrame.")
            return None

        numeric_df = df[columns].copy()

        if method not in NumericNormalizeMethod:
            raise ValueError(f"Invalid method: {method}")

        scaler = method.value[1]()
        scaled_values = scaler.fit_transform(numeric_df)

        normalized_df = pd.DataFrame(scaled_values, columns=numeric_df.columns, index=df.index)

        self.normalized_numeric_df = normalized_df

        if verbose:
            print("=" * 60)
            print("NUMERIC NORMALIZATION COMPLETED")
            print("=" * 60)
            print(f"Method used : {method}")
            print(
                f"Columns      : "
                f"{len(columns)}"
            )
            print(columns)
            print("=" * 60)

        return normalized_df

    def extract_normalized_categorical_data(
            self, method: CategoricalNormalizeMethod = CategoricalNormalizeMethod.ONEHOT,
            columns: list[str] | None = None, verbose: bool = True
    ) -> pd.DataFrame | None:
        """
        Extracts and normalizes categorical data from the dataframe based on the specified normalization method. The
        categorical columns are identified either automatically or from the provided list of columns, and the selected
        columns are transformed using the specified normalization method. The resulting transformed dataframe is returned
        and optionally logged in a verbose mode.

        :param method: The method to use for normalizing categorical data. Acceptable values are those defined in the
            CategoricalNormalizeMethod enum.
        :type method: CategoricalNormalizeMethod

        :param columns: A list of column names to normalize. If None, all categorical columns in the dataframe are selected.
            Defaults to None.
        :type columns: list[str] | None

        :param verbose: Indicates whether to print detailed logs of the normalization process. Defaults to True.
        :type verbose: bool

        :return: A dataframe containing the normalized categorical data or None if the dataframe is not ready.
        :rtype: pd.DataFrame | None
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        if columns is None:
            columns = df.columns.difference(df.select_dtypes(include=["number"]).columns).tolist()

        columns = [col for col in columns if col in df.columns]

        cat_df = df[columns].copy()

        encoded_df = pd.DataFrame(index=df.index)

        if method not in CategoricalNormalizeMethod:
            raise ValueError(f"Invalid method: {method}")

        if method == CategoricalNormalizeMethod.ONEHOT:
            encoded_df = pd.get_dummies(
                cat_df,
                drop_first=False
            )
        if method == CategoricalNormalizeMethod.ORDINAL:
            for col in columns:
                uniques = cat_df[col].astype(str).fillna("Unknown").unique()
                mapping = {
                    val: idx
                    for idx, val in enumerate(uniques)
                }
                encoded_df[col] = cat_df[col].astype(str).map(mapping)
        if method == CategoricalNormalizeMethod.UNIFORM:
            for col in columns:
                uniques = cat_df[col].astype(str).fillna("Unknown").unique()
                n = len(uniques)
                mapping = {
                    val: i / (n - 1 if n > 1 else 1)
                    for i, val in enumerate(uniques)
                }
                encoded_df[col] = cat_df[col].astype(str).map(mapping)

        self.normalized_categorical_df = encoded_df

        if verbose:
            print("=" * 60)
            print("CATEGORICAL ENCODING COMPLETED")
            print("=" * 60)
            print(f"Method used : {method}")
            print(f"Columns     : {columns}")
            print(f"Output shape: {encoded_df.shape}")
            print("=" * 60)

        return encoded_df

    def merge_normalized_data(
            self,
            merge_parts: list[pd.DataFrame] | None = None,
            verbose: bool = True
    ) -> pd.DataFrame | None:
        """
        Merge normalized numeric and categorical data into a single DataFrame.

        This method combines various normalized data parts (numeric and/or
        categorical) into a single DataFrame while ensuring no duplicate
        columns persist in the resulting data. It provides an option
        for verbose output to summarize the merging process and resulting
        shape and feature counts. The merged dataframe is stored within
        the object for further use.

        :param merge_parts: List of pandas DataFrame objects to be merged.
            If None, attempts to use the object's internal normalized
            numeric and categorical DataFrames for merging.
        :param verbose: Boolean flag to enable or disable logging detailed
            information about merging completion, shape, and feature
            counts.
        :return: A pandas DataFrame containing the merged numeric and
            categorical features, or None if no data is available
            for merging.
        """

        if merge_parts is None:
            merge_parts = []

            if not self.df_ready():
                return None

            numeric_df = getattr(
                self,
                "normalized_numeric_df",
                None
            )
            categorical_df = getattr(
                self,
                "normalized_categorical_df",
                None
            )
            if numeric_df is None and categorical_df is None:
                print(
                    "No normalized data found. "
                    "Run feature extraction first."
                )
                return None
            if numeric_df is not None:
                merge_parts.append(numeric_df)
            if categorical_df is not None:
                merge_parts.append(categorical_df)
        else:
            if not isinstance(merge_parts, list):
                raise ValueError("merged_parts must be a list of DataFrames")
            for part in merge_parts:
                if not isinstance(part, pd.DataFrame):
                    raise ValueError("Each item in merged_parts must be a DataFrame")

        merged_df = pd.concat(
            merge_parts,
            axis=1
        )
        merged_df: pd.DataFrame = merged_df.loc[
            :, ~merged_df.columns.duplicated()
        ]

        self.merged_normalized_df = merged_df

        if verbose:
            print("=" * 60)
            print("FEATURE MERGING COMPLETED")
            print("=" * 60)
            print(f"Shape: {merged_df.shape}")
            for merged_part in merge_parts:
                is_numeric = merged_part.dtypes.apply(lambda x: x.kind == "f").any()
                print(
                    f"{'Numeric' if is_numeric else 'Categorical'} features : "
                    f"{0 if merged_part.empty else merged_part.shape[1]}"
                )
            print("=" * 60)

        return merged_df

    def plot_all_associations_heatmap(
            self, show_matrix: bool = True, show_plot: bool = True, return_figure: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, go.Figure] | None:
        """
        Generates a heatmap of associations between features in a DataFrame.

        This method calculates association strengths between all pairs of columns
        in the DataFrame, creating a symmetric matrix where rows and columns correspond
        to features and the values represent the degree of association. The calculation
        logic varies depending on the types of data (numerical or categorical) and uses
        methods like Pearson correlation, chi-squared test, or variance analysis. The
        resulting association matrix can be displayed as a table, a heatmap, or returned
        as an output.

        :param show_matrix: Determines whether to display the association matrix table in the console.
        :param show_plot: Decides whether to render and display the heatmap of the association matrix.
        :param return_figure: Indicates whether to return the association matrix along with the plotly figure.
        :return: If `return_figure` is True, returns a tuple containing the association matrix
                 as a pandas.DataFrame and the heatmap figure as a plotly.graph_objects.Figure.
                 Otherwise, returns only the association matrix as a pandas.DataFrame or None
                 if the DataFrame is not ready.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        cols = df.columns.tolist()
        n_cols = len(cols)

        assoc_matrix = pd.DataFrame(
            np.zeros((n_cols, n_cols)),
            index=cols, columns=cols
        )

        is_numeric = {
            col: pd.api.types.is_numeric_dtype(df[col])
            for col in cols
        }

        for i in range(n_cols):
            for j in range(i, n_cols):

                col1 = cols[i]
                col2 = cols[j]

                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue

                valid_data = df[[col1, col2]].dropna()

                if len(valid_data) < 2:
                    val = 0.0
                else:
                    is_num1 = is_numeric[col1]
                    is_num2 = is_numeric[col2]

                    if is_num1 and is_num2:
                        x = valid_data[col1]
                        y = valid_data[col2]
                        val = abs(x.corr(y, method="pearson")) if x.nunique() <= 1 or y.nunique() <= 1 else 0.0
                    elif not is_num1 and not is_num2:
                        confusion_matrix = pd.crosstab(valid_data[col1], valid_data[col2])

                        if confusion_matrix.empty or min(confusion_matrix.shape) <= 1:
                            val = 0.0
                        else:
                            chi2 = chi2_contingency(confusion_matrix)[0]

                            n = confusion_matrix.sum().sum()

                            r, c = confusion_matrix.shape
                            k = min(r - 1, c - 1)

                            val = (
                                np.sqrt(chi2 / (n * k))
                                if n > 0 and k > 0
                                else 0.0
                            )
                    else:

                        cat_col, num_col = (
                            (col1, col2)
                            if not is_num1
                            else (col2, col1)
                        )

                        categories = (
                            valid_data[cat_col]
                            .dropna()
                            .unique()
                        )

                        if len(categories) <= 1:
                            val = 0.0
                        else:
                            groups = [
                                valid_data[
                                    valid_data[cat_col] == c
                                ][num_col]
                                for c in categories
                            ]

                            groups = [g for g in groups if len(g) > 0]

                            grand_mean = valid_data[num_col].mean()

                            ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()

                            ss_between = sum(
                                len(g) * (g.mean() - grand_mean) ** 2
                                for g in groups
                            )

                            val = (
                                np.sqrt(ss_between / ss_total)
                                if ss_total > 0
                                else 0.0
                            )

                assoc_matrix.loc[col1, col2] = round(val, 3)
                assoc_matrix.loc[col2, col1] = round(val, 3)

        if show_matrix:
            print("=" * 70)
            print("ASSOCIATION MATRIX")
            print("=" * 70)
            display(assoc_matrix)
            print("=" * 70)

        fig = px.imshow(
            assoc_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="viridis",
            zmin=0,
            zmax=1,
            title=(
                "<b>Feature Association Heatmap</b><br><sup>Values represent the strength of association between features, "
                "with 1 indicating perfect association and 0 indicating no association.</sup>"
            ),
            labels=dict(
                color="Association Strength"
            )
        )

        fig.update_layout(
            height=max(500, n_cols * 45),
            width=max(650, n_cols * 45),
            template="plotly_white"
        )

        if show_plot:
            fig.show()

        if return_figure:
            return assoc_matrix, fig

        return assoc_matrix


class AutoTypeCorrector(Scaler):
    """
    Class AutoTypeCorrector

    Description:
        This class performs automatic type correction on a given DataFrame by transforming
        non-numeric columns to numeric data types based on specified patterns and formats.

        The primary purpose of this class is to streamline the process of converting
        columns containing numeric data stored as strings into proper numeric types.
        It also provides comprehensive statistics about the conversion process for
        further analysis.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms a DataFrame by attempting to convert its non-numeric columns to numeric format.
        verbose: Provides detailed information about the results of an automatic type correction process.

    :ivar numeric_formats: A dictionary of user-defined replacement patterns for
        formatting conversions in addition to the predefined patterns.
    :type numeric_formats: dict[str, str] | None
    """

    NUMERIC_FORMATS = {
        ",": "",
        "$": "",
        "%": ""
    }
    _last_stat: dict[str, Any] = {}

    def __init__(self, numeric_formats: dict[str, str] | None = None):
        """
        Initializes an instance of the class with optional numeric formats.

        :param numeric_formats: A dictionary where the keys are string identifiers for
            numeric formats and the values are string representations of the corresponding
            format. Defaults to None.
        :type numeric_formats: dict[str, str] | None
        """
        self.numeric_formats = numeric_formats

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the provided data.

        This method uses the provided dataset to train the model and adjust its
        internal parameters accordingly. It is called with input features and,
        optionally, target values when applicable. The method modifies the
        state of the model and returns the object itself to support chaining.

        :param X: The input features for training the model. Expected to be
            a pandas DataFrame containing the relevant data.
        :param y: (Optional) The target values corresponding to the input data.
            Expected to be a pandas Series or None if the method does not require
            a target variable during fitting.
        :return: The instance of the class itself after fitting to the given data.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a DataFrame by attempting to convert its non-numeric columns to numeric format based on specified
        replacement patterns. Updates columns with a successful conversion rate above 50% and skips others. Stores
        conversion statistics for later inspection.

        :param X: The input DataFrame to be transformed.
        :type X: pd.DataFrame
        :return: A transformed DataFrame with certain columns converted to numeric types, based on the specified
            replacement patterns and conversion rules.
        :rtype: pd.DataFrame
        """
        df = X.copy()

        converted_columns = []
        skipped_columns = []

        all_formats = self.NUMERIC_FORMATS | (self.numeric_formats or {})

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                continue

            cleaned_column = df[column].astype(str)
            for format_char, replacement in all_formats.items():
                is_regex = (
                        isinstance(format_char, str)
                        and format_char.startswith("/")
                        and format_char.endswith("/")
                )
                pattern = format_char[1:-1] if is_regex else format_char
                cleaned_column = cleaned_column.str.replace(pattern, replacement, regex=is_regex)

            converted = pd.Series(pd.to_numeric(cleaned_column, errors="coerce"))

            if converted.notna().any():
                original_non_null = int(df[column].notna().sum())
                converted_non_null = int(converted.notna().sum())
                conversion_rate = converted_non_null / original_non_null if original_non_null != 0 else 0

                if conversion_rate > 0.5:
                    df[column] = converted
                    converted_columns.append(column)
                else:
                    skipped_columns.append(column)
            else:
                skipped_columns.append(column)

        X = df

        self._last_stat = {
            "converted_columns": converted_columns,
            "skipped_columns": skipped_columns,
            "data_types": X.dtypes
        }

        return X

    def verbose(self):
        """
        Provides detailed information about the results of an automatic type correction
        process by printing the converted columns, skipped columns, and updated
        data types.

        :return: None
        """
        converted_columns = self._last_stat.get("converted_columns", [])
        skipped_columns = self._last_stat.get("skipped_columns", [])
        data_types = self._last_stat.get("data_types", {})

        print("=" * 60)
        print("AUTO TYPE CORRECTION COMPLETED")
        print("=" * 60)

        print("\nConverted Columns:", f"\n{converted_columns}" if converted_columns else "N/A")
        print("\nSkipped Columns:", f"\n{skipped_columns}" if skipped_columns else "N/A")
        print("\nUpdated Data Types:",
              f"\n{data_types}" if data_types is not None and not data_types.empty else "N/A")

        print("=" * 60)


class ColumnRemover(Scaler):
    """
    class ColumnRemover

    Description:
        Summary of what the class does.

        Handles the removal of specific columns from a pandas DataFrame, either programmatically
        using a predefined list of column names or interactively via user input. Tracks metadata
        about the transformation process for logging and debugging purposes.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms the input DataFrame by removing specified columns.
        verbose: Provides detailed information about the column deletion process.

    :ivar columns: A list of column names to be removed from DataFrames. If not provided,
        this attribute defaults to None, and columns can be selected interactively.
    :type columns: list[str]
    """

    _last_stat: dict[str, int | list[str]] = {}
    columns: list[str] = NotImplemented

    def __init__(self, columns: list[str] | None = None) -> None:
        """
        Initializes the instance with a list of column names.

        This constructor accepts an optional list of column names that can be used
        to configure the instance. If no list is provided, the columns attribute
        will be initialized to None.

        :param columns: A list of column names to configure the instance.
        :type columns: list[str] | None
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model using the provided input data and optionally the target labels.

        This method standardizes the process of training a model by using
        training features and optional target values. It modifies the
        internal state of the model based on the input data, allowing it
        to be used for predictions or further evaluations.

        :param X: The input data as a pandas DataFrame, containing the
            features needed to train the model.
        :param y: Optional target labels as a pandas Series. If provided,
            these labels are used to optimize the training process. Defaults
            to None if no labels are provided.
        :return: The instance of the model itself (`self`) after fitting
            it with the input data.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes specified columns from a Pandas DataFrame. If no columns are defined, prompts the user to select columns
        to remove interactively. Maintains metadata about the transformation, including the list of columns removed
        and the DataFrame's shape before and after the transformation.

        :param X: Input DataFrame to be transformed.
        :type X: pd.DataFrame
        :return: Transformed DataFrame with specified columns removed.
        :rtype: pd.DataFrame
        """
        df = X.copy()

        if self.columns is None:
            columns = df.columns.tolist()
            print("Available columns:")
            print("\n".join(f"{index + 1}. {column}" for index, column in enumerate(columns)))
            try:
                user_input = input("Enter the column numbers to remove (comma-separated), or 'q' to quit: ")

                if user_input.lower() == "q":
                    return df

                selected_indices = [int(num.strip()) - 1 for num in user_input.split(",")]
                self.columns = [columns[index] for index in selected_indices if 0 <= index < len(columns)]
            except ValueError:
                raise ValueError("Invalid input. Please enter valid column numbers.")

        cols_before = df.shape[1]

        valid_cols = [
            col for col in self.columns
            if col in df.columns
        ]

        df = df.drop(columns=valid_cols)

        cols_after = df.shape[1]

        X = df

        self._last_stat = {
            "columns": self.columns,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "valid_cols": valid_cols
        }

        return X

    def verbose(self):
        """
        Outputs detailed logs regarding the column deletion process.

        This method provides a verbose summary of the column deletion operation,
        displaying information such as the requested columns, valid (deleted) columns,
        and the difference between the number of columns before and after the process.

        :return: None
        """
        cols_before = self._last_stat.get("cols_before", 0)
        cols_after = self._last_stat.get("cols_after", 0)

        print("=" * 60)
        print("COLUMN DELETION COMPLETED")
        print("=" * 60)
        print(
            f"Columns requested : "
            f"{self._last_stat.get('columns', 'N/A')}"
        )
        print(
            f"Columns deleted   : "
            f"{self._last_stat.get('valid_cols', 'N/A')}"
        )
        print(
            f"Columns removed   : "
            f"{cols_before - cols_after if cols_before and cols_after else 'N/A'}"
        )
        print("=" * 60)


class DiscardMethod(Enum):
    FIRST = ("first", "first")
    LAST = ("last", "last")
    ALL = ("all", False)


class DuplicateRemover(Scaler):
    """
    Class DuplicateRemover

    Description:
        Duplicates remover within a dataset.

        This class extends `Scaler` and provides functionality to identify and
        remove duplicate rows from a dataset. The removal process can be
        customized using different discard strategies and column subsets. The
        class also maintains statistics about the duplicates removed during the
        transformation process.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms the input DataFrame by removing duplicate rows.
        verbose: Provides detailed information about the duplicate removal process.

    :ivar discardMethods: The discard strategies available for resolving duplicates.
    :type discardMethods: DiscardMethod
    :ivar subset: List of column names to use for duplicate detection. Applies to
                  all columns if left unmodified or set to None.
    :type subset: list[str]
    :ivar keep: The discard strategy to apply when removing duplicates. This
                must be a valid option from the DiscardMethod enumeration.
    :type keep: DiscardMethod
    """

    discardMethods = DiscardMethod
    _last_stat: dict[str, float] = {}
    subset: list[str] = NotImplemented
    keep: DiscardMethod = NotImplemented

    def __init__(self, subset: list[str] | None = None, keep: DiscardMethod = DiscardMethod.FIRST):
        """
        Initializes an instance of the class with specified subset and discard handling method.

        :param subset: A list of column names to operate on. If None, applies to all columns.
        :param keep: A value of type DiscardMethod representing the method to retain certain rows
            during processing. Must be one of the valid options in the DiscardMethod enumeration.

        :raises ValueError: If the provided `keep` argument is not an instance of DiscardMethod.
        """
        self.subset = subset
        self.keep = keep

        if self.keep not in self.discardMethods:
            raise ValueError("keep must be of type DiscardMethod")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the given data.

        This method trains the model using the provided feature matrix and
        optional target values. It does not modify the input data.

        :param X: Feature matrix to train the model.
        :type X: pd.DataFrame
        :param y: Target values corresponding to the feature matrix. This
                  parameter is optional if the model does not require target
                  values for training.
        :type y: pd.Series | None
        :return: The instance of the class itself after fitting the model.
        :rtype: Self
        """
        return self

    def transform(self, X) -> Self:
        """
        Transforms the provided DataFrame by removing duplicate rows based on the specified subset
        and keep policy. Updates internal statistics about the operation, including the number
        of duplicates found, rows removed, and final row count.

        :param X: The input DataFrame to transform.
        :type X: pandas.DataFrame

        :return: A transformed DataFrame with duplicates removed.
        :rtype: Self
        """
        df = X.copy()

        rows_before = len(df)

        duplicate_count = df.duplicated(
            subset=self.subset
        ).sum()

        df = df.drop_duplicates(
            subset=self.subset, keep=self.keep.value[1]  # type: ignore : the second element is either "first", "last" or False
        )

        rows_after = len(df)
        removed_rows = rows_before - rows_after

        X = df

        self._last_stat = {
            "duplicate_count": duplicate_count,
            "removed_rows": removed_rows,
            "rows_after": rows_after,
            "subset": self.subset.copy() if self.subset else None,
            "keep": self.keep.name
        }

        return X

    def verbose(self):
        """
        Prints a detailed summary of the duplicate removal process to the console.

        This method outputs relevant statistics about the duplicate removal operation,
        including the number of duplicates found, rows removed, rows remaining, as well
        as details about the subset of columns and the strategy used to handle duplicates.

        :return: None
        """
        print("=" * 60)
        print("DUPLICATE REMOVAL COMPLETED")
        print("=" * 60)
        print(f"Duplicates found : {self._last_stat.get('duplicate_count', 'N/A')}")
        print(f"Rows removed     : {self._last_stat.get('removed_rows', 'N/A')}")
        print(f"Rows remaining   : {self._last_stat.get('rows_after', 'N/A')}")
        if subset := self._last_stat.get('subset', None) is not None:
            print(f"Subset columns   : {subset}")
        print(f"Keep strategy    : {self._last_stat.get('keep', 'N/A')}")
        print("=" * 60)


class Strategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"


class MissingValueHandler(Scaler):
    """
    Class MissingValueHandler

    Description:
        Handles missing values in datasets by applying various imputation strategies.

        This class is designed to process missing values in a dataset, using user-specified
        imputation strategies such as replacing missing values with the mean, median, mode,
        or a constant value. The class supports operations on selected columns or the entire
        dataset. It provides methods for fitting the imputation strategy to a dataset and
        applying the transformation to handle missing values.

    Methods:
        fit: Prepares the imputation strategy based on the provided dataset, allowing for any necessary calculations or state initialization.
        transform: Imputes missing values in the specified columns of the input DataFrame using the selected strategy.
        verbose: Provides detailed information about the imputation process, including the number of missing values before and after the operation.

    :ivar strategies: The predefined list of valid strategies for handling missing values.
    :type strategies: Strategy
    :ivar strategy: The selected strategy for handling missing values.
    :type strategy: Strategy
    :ivar fill_value: The constant value to be used when the strategy is CONSTANT.
    :type fill_value: float | None
    :ivar columns: The list of columns to which the defined strategy is applied. If None,
                   the strategy is applied to all columns in the dataset.
    :type columns: list[str] | None
    """
    strategies = Strategy
    strategy: Strategy = NotImplemented
    fill_value: float | None = NotImplemented
    columns: list[str] | None = NotImplemented
    _last_stat: dict[str, float | str] = {}

    def __init__(self, strategy: Strategy = Strategy.MEAN, fill_value=None, columns: list[str] | None = None) -> None:
        """
        Initializes an object with specified strategy, fill value, and columns. The strategy determines
        how missing values should be handled, whether by using a mean, constant value, or other pre-defined
        options. For the `CONSTANT` strategy, a `fill_value` must always be provided. If the specified
        strategy is not valid, an exception is raised.

        :param strategy: The strategy to use for handling missing values.
        :type strategy: Strategy
        :param fill_value: The value to use when the strategy is CONSTANT. It must be provided if `strategy`
                           is CONSTANT.
        :type fill_value: Any
        :param columns: A list of column names on which to apply the strategy, or None to apply to all columns.
        :type columns: list[str] | None

        :raises ValueError: If `fill_value` is None while using the CONSTANT strategy.
        :raises ValueError: If the provided strategy is not in the predefined list of valid strategies.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

        if self.strategy == Strategy.CONSTANT and self.fill_value is None:
            raise ValueError("Constant strategy requires a fill_value.")

        if strategy not in self.strategies:
            raise ValueError(f"Invalid strategy: {strategy}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the given data. The method takes input features and optionally
        target labels to train the model or prepare it for further predictions or
        operations.

        :param X: A DataFrame containing the input features for training.
        :param y: An optional Series containing the target labels. Defaults to None.
        :return: The instance of the fitted model itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in specified columns of a Pandas DataFrame based on a given strategy.

        This method applies various imputation strategies, such as mean, median, mode, or constant,
        to the columns in the provided DataFrame. If no columns are specified, all columns of the
        DataFrame are processed. The method also tracks and updates missing value statistics before
        and after the imputation process.

        :param X: The input DataFrame containing the data to transform.
        :type X: pd.DataFrame

        :return: A DataFrame with missing values replaced in the specified or applicable columns.
        :rtype: pd.DataFrame
        """
        df = X.copy()

        if self.columns is None:
            self.columns = df.columns.tolist()

        missing_before = df.isna().sum().sum()

        for col in self.columns:
            if col not in df.columns:
                print(f"Column {col} not found in the DataFrame, Skipping.")
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if self.strategy == Strategy.MEAN:
                    df[col] = df[col].fillna(df[col].mean())
                elif self.strategy == Strategy.MEDIAN:
                    df[col] = df[col].fillna(df[col].median())
                elif self.strategy == Strategy.MODE:
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif self.strategy == Strategy.CONSTANT:
                    df[col] = df[col].fillna(self.fill_value)
                else:
                    raise ValueError(f"Invalid strategy: {self.strategy}")
            else:
                if self.strategy == Strategy.MODE:
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif self.strategy == Strategy.CONSTANT:
                    df[col] = df[col].fillna(self.fill_value)
                else:
                    raise ValueError(f"Invalid strategy: {self.strategy}")

        missing_after = df.isna().sum().sum()
        new_missing = missing_after - missing_before

        X = df

        self._last_stat = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "new_missing": new_missing,
            "strategy": self.strategy.value,
        }

        return X

    def verbose(self):
        """
        Prints a detailed summary of the missing value imputation process, including
        statistics before and after the operation, new missing values detected, and
        the strategy used for imputation.

        :return: None
        :rtype: NoneType
        """
        print("=" * 60)
        print("MISSING VALUE IMPUTATION COMPLETED")
        print("=" * 60)
        print(f"Missing before : {self._last_stat.get('missing_before', 'N/A')}")
        print(f"Missing after  : {self._last_stat.get('missing_after', 'N/A')}")
        print(f"New missing detected  : {self._last_stat.get('new_missing', 'N/A')}")
        print(f"Strategy used : {self._last_stat.get('strategy', 'N/A')}")
        print("=" * 60)


class MissingValueSanitizer(Scaler):
    """
    Class MissingValueSanitizer

    Description:
        Defines a class for sanitizing missing values in a DataFrame by identifying and
        replacing patterns indicative of missing data with a standardized representation.

        This class allows customization of missing value patterns, enabling users to define
        their own rules in addition to predefined patterns. The sanitization process ensures
        consistent handling of missing data, particularly for data cleaning and preprocessing tasks.

    Methods:
        fit: Fits the model to the provided data, allowing for any necessary setup or state initialization.
        transform: Transforms a DataFrame by replacing specified missing patterns with pandas' native missing value representation.
        verbose: Provides detailed information about the missing value sanitization process.

    :ivar custom_missing_patterns: A list of user-defined patterns treated as missing values.
    :type custom_missing_patterns: list[str] | None
    """

    MISSING_VALUES_PATTERNS = [
        "?",
        "n/a",
        "N/A",
        "na",
        "NA",
        "null",
        "NULL",
        "None",
        "none",
        "",
        " "
    ]
    _last_stat: dict[str, float] = {}

    def __init__(self, custom_missing_patterns: list[str] | None = None) -> None:
        """
        Initializes the class with an optional list of custom missing patterns.

        The constructor sets up the `custom_missing_patterns` attribute, which can be used
        to define specific patterns that are recognized as indicating missing values.

        :param custom_missing_patterns: A list of strings representing user-defined patterns
            that denote missing values. Defaults to None, in which case no custom patterns are
            defined.
        """
        self.custom_missing_patterns = custom_missing_patterns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model using the provided input data and labels.

        This method sets up the model based on the provided features and optional
        target labels. The method allows for creating and customizing any internal
        model state required for further predictions or transformations.

        :param X: Input data used for fitting the model. Must be provided as a
            pandas DataFrame.
        :param y: Optional target labels corresponding to the input data. Must
            be provided as a pandas Series, or `None` if not applicable.
        :return: Returns the fitted instance of the model.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a given DataFrame by identifying and replacing values specified in missing patterns
        with `pd.NA`. This approach allows handling and standardizing missing data representations.

        :param X: The input DataFrame that needs transformation. The DataFrame is expected to contain
            columns where missing values might be represented by patterns specified in the
            `MISSING_VALUES_PATTERNS` as well as any user-defined patterns.

        :return: A transformed DataFrame where values matching specified missing patterns are replaced
            with pandas' native missing value representation (`pd.NA`).
        :rtype: pd.DataFrame
        """
        all_patterns = list(set(self.MISSING_VALUES_PATTERNS + (self.custom_missing_patterns or [])))

        df = X.copy()
        missing_before = df.isnull().sum().sum()

        df = df.map(lambda x: pd.NA if isinstance(x, str) and x.strip() in all_patterns else x)
        df.replace(all_patterns, pd.NA, inplace=True)

        missing_after = df.isnull().sum().sum()

        new_missing = missing_after - missing_before

        X = df

        self._last_stat = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "new_missing": new_missing
        }

        return X

    def verbose(self):
        """
        Prints verbose information about the missing value sanitization process.

        Detailed summary of missing value statistics is provided, including missing
        values count before and after sanitization, as well as any newly detected
        missing values during the process.

        :print: Directly outputs the missing value statistics to the standard output.
        """
        print("=" * 60)
        print("MISSING VALUE SANITIZATION COMPLETED")
        print("=" * 60)
        print(f"Missing values before : {self._last_stat.get('missing_before', 'N/A')}")
        print(f"Missing values after  : {self._last_stat.get('missing_after', 'N/A')}")
        print(f"New missing detected  : {self._last_stat.get('new_missing', 'N/A')}")
        print("=" * 60)


class OutlierMethod(Enum):
    CAP = "cap"
    REMOVE = "remove"


class OutlierHandler(Scaler):
    """
    Class OutlierHandler

    Description:
        Handles outlier removal or capping within datasets.

        This class is designed to process numerical columns in datasets and either cap or remove
        outliers based on the Interquartile Range (IQR). It offers configurable options for the
        columns to be processed, the outlier handling method, and a factor influencing the
        thresholds. It is meant to enhance data preprocessing workflows by standardizing
        outlier-handling techniques.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms the input DataFrame by handling outliers.
        verbose: Provides detailed information about the outlier handling process.

    :ivar methods: Set of strategies supported for handling outliers. Includes 'CAP'
        for capping outliers within IQR bounds and 'REMOVE' for removing rows
        containing outliers.
    :type methods: OutlierMethod
    :ivar columns: List of column names to apply the outlier handling operation on.
        If not specified, it defaults to processing all numeric columns.
    :type columns: list[str]
    :ivar method: The strategy to handle outliers. Must be either 'CAP' or 'REMOVE'.
    :type method: OutlierMethod
    :ivar factor: Multiplier applied to the IQR for computing thresholds to identify
        outliers.
    :type factor: float
    """
    methods = OutlierMethod
    _last_stat: dict[str, float | str] = {}
    columns: list[str] = NotImplemented
    method: OutlierMethod = NotImplemented
    factor: float = NotImplemented

    def __init__(
            self, columns: list[str] | None = None, method: OutlierMethod = OutlierMethod.CAP, factor: float = 1.5
    ):
        """
        Initializes an instance with specified configuration for handling outliers
        in datasets. This constructor allows specifying the columns to be processed,
        the method of outlier handling, and a factor influencing the threshold logic.

        :param columns: Specific list of column names on which to apply the outlier
            operation. If None, the operation will be applied to all columns.
        :param method: The strategy to handle outliers. Supported options are 'CAP',
            which caps outliers to specified thresholds, or 'REMOVE', which removes
            identified outliers from the dataset.
        :param factor: A numerical value that indicates the threshold multiplier
            for determining what constitutes an outlier. This is used in conjunction
            with the specified `method`.
        :raises ValueError: If the `method` provided is not one of the supported
            options ('CAP' or 'REMOVE').
        """
        self.columns = columns
        self.method = method
        self.factor = factor

        if self.method not in self.methods:
            raise ValueError("method must be 'CAP' or 'REMOVE'")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the provided input data.

        This method allows the model to learn from the provided features (`X`) and
        optionally, the target values (`y`). If `y` is provided, it is utilized in
        training; otherwise, the method relies solely on the data in `X`. The method
        returns the instance of the class to facilitate method chaining.

        :param X: The input data as a DataFrame which the model should learn from.
        :param y: The target values as a Series, or None if there are no target values.
        :return: The fitted instance of the class itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by handling outliers based on the specified method
        and factor using the Interquartile Range (IQR). The transformation either caps
        outliers within the IQR bounds or removes rows entirely based on the chosen method.

        :param X: Input DataFrame to be transformed.

        :return: Transformed DataFrame with outliers handled as per configuration.
        :rtype: pd.DataFrame
        """
        df = X.copy()

        if self.columns is None:
            self.columns = df.select_dtypes(include=["number"]).columns.tolist()

        total_outliers = 0

        for col in self.columns:
            if col not in df.columns:
                print(f"Column {col} not found in the DataFrame, Skipping.")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column {col} is not numeric, Skipping.")
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)]
            total_outliers += len(outliers)

            if self.method == self.methods.CAP:
                df[col] = df[col].clip(lower, upper)
            elif self.method == self.methods.REMOVE:
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            else:
                raise ValueError("method must be 'CAP' or 'REMOVE'")

        X = df

        self._last_stat = {
            "total_columns": len(self.columns),
            "total_outliers": total_outliers,
            "method": self.method.value,
        }

        return X

    def verbose(self):
        """
        Prints a detailed summary of the outlier handling process, including statistics about
        the columns processed, the number of outliers detected, and the method used.

        :return: None
        """
        print("=" * 60)
        print("OUTLIER HANDLING COMPLETED")
        print("=" * 60)
        print(f"Columns processed : {self._last_stat.get('total_columns', 'N/A')}")
        print(f"Outliers detected : {self._last_stat.get('total_outliers', 'N/A')}")
        print(f"Method            : {self._last_stat.get('method', 'N/A')}")
        print("=" * 60)


class RowRemover(Scaler):
    """
    class RowRemover

    Description:
        Class responsible for removing rows from a pandas DataFrame.

        The `RowRemover` class provides functionality to remove specified rows from a DataFrame.
        It supports user-defined row indices and allows dynamic input of rows if not pre-specified.
        The class includes methods for fitting data, transforming the input DataFrame by deleting
        rows, and displaying verbose statistics about the deletion process.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms the input DataFrame by removing specified rows.
        verbose: Provides detailed information about the row deletion process.

    :ivar rows: A list of integers representing the indices of rows to be deleted.
    :type rows: list[int] | None
    """

    _last_stat: dict[str, int] = {}
    rows: list[int] = NotImplemented

    def __init__(self, rows: list[int] | None = None) -> None:
        """
        Initializes a new instance of the class.

        This constructor is responsible for setting up the 'rows' attribute
        for the newly created instance. The 'rows' attribute is expected to be
        either a list of integers or None, providing flexibility in input data.

        :param rows: A list of integers representing the rows or None if no
            rows are provided.
        :type rows: list[int] | None
        """
        self.rows = rows

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the provided data.

        This method is used to train the model by fitting it to the input data (`X`) and
        optional target labels (`y`). The target labels can be omitted if the model
        does not require supervised learning. The method returns the instance of the
        model itself, which allows for method chaining.

        :param X: Input data for training. The data must be provided as a pandas
            DataFrame.
        :param y: Target labels for supervised learning tasks. This parameter is
            optional and can be set to None for unsupervised learning tasks. The labels
            must be provided as a pandas Series or None.
        :return: The instance of the fitted model (self).
        :rtype: Self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Deletes specified rows from the input DataFrame based on the user-provided or predefined list
        of row indices. If no indices are provided beforehand, prompts the user to input them,
        handles both single and range row indices, validates indices against the DataFrame, and
        only deletes valid rows.

        :param X: The input DataFrame to be transformed by removing specified rows.
        :type X: pd.DataFrame
        :return: A new DataFrame with the specified rows removed.
        :rtype: pd.DataFrame
        :raises ValueError: If the user input for row indices is not in a valid format.
        """

        df = X.copy()

        if self.rows is None:
            print(
                "Here's how to enter the row indices to delete:"
                "\n\tEg:- 1, 3, 5-7"
                "\n\tThis will delete rows 1, 3, and 5 to 7 (inclusive)."
            )
            try:
                user_input = input("Enter the row indices to delete, enter 'q' to quit: ")

                if user_input.lower() == "q":
                    return df

                rows = []
                for row in user_input.split(","):
                    if "-" in row:
                        start, end = map(int, row.split("-"))
                        rows.extend(range(start, end + 1))
                    else:
                        rows.append(int(row))
                self.rows = rows
            except ValueError:
                raise ValueError("Invalid input. Please enter a valid list of row indices.")

        rows_before = len(df)
        valid_rows = [row for row in self.rows if 0 <= row < rows_before]

        df = df.drop(valid_rows)

        rows_after = len(df)

        X = df

        self._last_stat = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "valid_rows": valid_rows,
            "rows": self.rows
        }

        return X

    def verbose(self):
        """
        Provides functionality to display verbose output for row deletion statistics.

        This method outputs formatted information about the number of rows initially requested
        for deletion, the number of valid rows successfully deleted, and the net number
        of rows removed from the system.

        :raises KeyError: If required keys are missing from the internal state tracking the row
            deletion statistics.
        :return: None
        """
        rows_before = self._last_stat.get("rows_before", 0)
        rows_after = self._last_stat.get("rows_after", 0)

        print("=" * 60)
        print("ROW DELETION COMPLETED")
        print("=" * 60)
        print(f"Rows requested : {self._last_stat.get('rows', 'N/A')}")
        print(f"Rows deleted   : {self._last_stat.get('valid_rows', 'N/A')}")
        print(
            f"Rows removed   : "
            f"{rows_before - rows_after if rows_before and rows_after else 'N/A'}"
        )
        print("=" * 60)


class Case(Enum):
    TITLE = "title"
    LOWER = "lower"
    UPPER = "upper"
    NONE = "none"


class StandardizeFormator(Scaler):
    """
    class StandardizeFormator

    Description:
        Standardizes and formats column data in a DataFrame.

        Aimed at cleaning and processing textual column data by:
        - Applying case transformations.
        - Cleaning numeric strings (e.g., removing commas, dollar signs, etc.).
        - Applying custom mappings for specific variations.
        - Handling user-defined or automatically detected string columns.

        Use this class to clean and harmonize text data in DataFrame column headers for
        further analysis or modeling.

    Methods:
        fit: Fits the model to the provided data.
        transform: Transforms the input DataFrame by applying specified text processing rules.
        verbose: Provides detailed information about the text processing process.

    :ivar columns: List of columns to process. If not provided, columns with 'object' type
        are automatically processed.
    :type columns: list[str]
    :ivar cases: Enumeration of case transformations (e.g., LOWER, UPPER, TITLE).
    :type cases: Case
    :ivar case: Desired case transformation to apply to column data.
    :type case: Case
    :ivar custom_mappings: Dictionary of custom transformation mappings for specified columns
        where key is the column name and value is the mapping dictionary.
    :type custom_mappings: dict[str, dict]
    :ivar clean_numeric_strings: Flag to indicate whether numeric strings require formatting
        (e.g., removing commas, dollar signs, etc.).
    :type clean_numeric_strings: bool
    """

    _last_stat: dict[str, str | list[str]] = {}
    columns: list[str] = NotImplemented
    cases = Case
    case: Case = NotImplemented
    custom_mappings: dict[str, dict] = NotImplemented
    clean_numeric_strings: bool = NotImplemented

    def __init__(
            self, columns: list[str] | None = None, case: Case = Case.TITLE,
            custom_mappings: dict[str, dict] | None = None, clean_numeric_strings: bool = True
    ):
        """
        Initializes the instance of the class with configuration for column names
        processing, case transformation, custom mappings, and numeric string
        cleaning.

        :param columns: List of column names to process. If `None`, all columns
            will be processed.
        :type columns: list[str] | None
        :param case: Case transformation to be applied to column names. This can
            be `Case.TITLE`, `Case.UPPER`, or `Case.LOWER`.
        :type case: Case
        :param custom_mappings: Custom mappings to apply to column names. This
            should be a dictionary where keys are column names and values are
            dictionaries containing mappings for transformations.
        :type custom_mappings: dict[str, dict] | None
        :param clean_numeric_strings: Indicates whether numeric strings should be
            cleaned (e.g., removing extra spaces or formatting artifacts).
        :type clean_numeric_strings: bool
        :raises ValueError: If the provided `case` is invalid.
        :raises ValueError: If the `clean_numeric_strings` parameter is not a
            boolean value.
        """
        self.columns = columns
        self.case = case
        self.custom_mappings = custom_mappings
        self.clean_numeric_strings = clean_numeric_strings

        if self.custom_mappings is None:
            self.custom_mappings = {}
        if self.case not in self.cases:
            raise ValueError(f"Invalid case: {self.case}")
        if self.clean_numeric_strings not in (True, False):
            raise ValueError(f"Invalid clean_numeric_strings: {self.clean_numeric_strings}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """
        Fits the model to the provided data.

        This method is responsible for training or adjusting the model parameters based
        on the input features and optional target values. It modifies the internal state
        of the model object as a result of the fitting process.

        :param X: Input features as a pandas DataFrame, where rows represent samples and
                  columns represent features.
        :param y: Optional target values as a pandas Series. If provided, it contains
                  the labels or outcomes corresponding to each row in X. If not
                  specified, the model may perform unsupervised learning or other tasks
                  that do not require targets.
        :return: The instance of the fitted model.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame based on the specified text processing rules.

        This method processes specified columns of the input DataFrame by performing operations
        such as trimming, case formatting, cleaning numeric strings, and applying custom mappings.
        If no columns are specified, it determines columns of type 'object' automatically to process.

        :param X: The input DataFrame to be transformed.
        :type X: pd.DataFrame
        :return: The transformed DataFrame after applying the text processing rules.
        :rtype: pd.DataFrame
        """

        df = X.copy()

        if self.columns in (None, NotImplemented):
            self.columns = (
                df.select_dtypes(include=["object", "str"])
                .columns
                .tolist()
            )

        processed_columns = []

        for col in self.columns:
            if col not in df.columns:
                print(f"Column {col} not found in the DataFrame, Skipping.")
                continue

            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
            if df[col].dtype == "str" or df[col].dtype == "object":
                if self.case == self.cases.LOWER:
                    df[col] = df[col].str.lower()
                elif self.case == self.cases.UPPER:
                    df[col] = df[col].str.upper()
                elif self.case == self.cases.TITLE:
                    df[col] = df[col].str.title()
                elif self.case == self.cases.NONE:
                    pass
                else:
                    raise ValueError("case must be of type Case or one of its values.")

                if self.clean_numeric_strings:
                    df[col] = (
                        df[col]
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace("%", "", regex=False)
                    )
                if col in self.custom_mappings:
                    mapping = {
                        str(k).strip().lower(): v
                        for k, v in self.custom_mappings[col].items()
                    }
                    df[col] = df[col].apply(
                        lambda x:
                        mapping.get(
                            str(x).lower(),
                            x
                        )
                        if pd.notna(x)
                        else x
                    )
                processed_columns.append(col)

        X = df

        self._last_stat = {
            "processed_columns": processed_columns,
            "case": self.case.name
        }

        return X

    def verbose(self):
        """
        Displays a detailed summary of the format standardization process. This
        function provides statistical and configuration details about the processed
        columns and the applied case formatting.

        :return: None
        :rtype: None
        """
        print("=" * 60)
        print("FORMAT STANDARDIZATION COMPLETED")
        print("=" * 60)
        print(
            f"Columns processed : "
            f"{len(self._last_stat.get('processed_columns', []))}"
        )
        print(self._last_stat.get("processed_columns", []))
        print(f"Case format      : {self._last_stat.get('case', 'N/A')}")
        print("=" * 60)
