import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency

__all__ = ["PlottingMethods"]


class PlottingMethods:
    """
    class PlottingMethods

    Description:
        Provides methods for data exploration and visualization on a given pandas DataFrame.

        This class contains methods for inspecting, validating, and visualizing the data
        in a pandas DataFrame. It includes features such as data readiness checks,
        univariate visualization of numeric columns, visualization of relationships
        between columns, and frequency distribution of categorical columns.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame used for analysis and visualization.

    Methods:
        get_methods_info: Retrieves information about public methods of the current instance.
        df_ready: Determines if the dataframe (df) has been loaded and is non-empty.
        plot_numeric_univariate: Generates a series of univariate visualizations for numeric columns.
        plot_relationship: Plots a visual representation of the relationship between two columns.
        plot_categorical_frequency: Generates a bar chart visualizing the frequency distribution of a categorical column.
        plot_all_associations_heatmap: Generates a heatmap visualizing the associations between all numerical and categorical columns.
        bar_chart: Generates an HTML representation of a bar chart visualizing the frequency of values in a column.
        pie_chart: Generate an HTML representation of a pie chart using Plotly based on the distribution of values in a column.
        histogram: Generates a histogram plot of a numeric column.
    """

    df: pd.DataFrame = NotImplemented

    def __init__(self, df: pd.DataFrame):
        """
        Initializes an object with a pandas DataFrame.

        :param df: The pandas DataFrame to be used for initialization.
        :type df: pd.DataFrame
        """
        self.df = df

    @classmethod
    def get_methods_info(cls):
        """
        Retrieves information about public methods of the current instance.

        Compiles a list of dictionaries containing details of the public methods
        implemented in the class instance. Each dictionary in the list includes
        the method name, its signature, and its docstring. Private, protected,
        and internal methods (those with names starting with an underscore) are excluded.

        Returns:
            list[dict]: A list of dictionaries where each dictionary represents
            a method with the following keys:
                - method (str): The name of the method.
                - signature (str): The signature of the method.
                - description (str): The docstring or "No description available" if no
                  docstring exists.
        """
        method_dicts = []
        methods = inspect.getmembers(cls, inspect.ismethod)

        for name, method in methods:
            if name.startswith('_'):
                continue
            signature = inspect.signature(method)
            docstring = method.__doc__
            formatted_docstring = docstring.strip() if docstring else "No description available"
            method_dicts += [{"method": name, "signature": str(signature), "description": formatted_docstring}]
        return method_dicts

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

    def plot_numeric_univariate(self, columns: list[str] | None = None):
        """
        Generates a series of univariate visualizations for numeric columns in the DataFrame.
        This includes a box plot, a scatter plot of index vs. values, and a histogram for
        each specified numeric column. If no `columns` are specified, the method automatically
        selects all numeric columns in the DataFrame for visualization.

        :param columns: A list of column names to generate plots for. If None, all numeric
                        columns in the DataFrame are used.
        :type columns: list[str] | None
        :return: A dictionary where the keys are column names and the values are plotly figure
                 objects corresponding to the univariate visualizations of those columns. If
                 the DataFrame is not ready or no valid columns are provided, None is returned.
        :rtype: dict[str, plotly.graph_objects.Figure] | None
        """
        if not self.df_ready():
            return None

        df = self.df.copy()

        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()

        figures = {}

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]

            fig = make_subplots(rows=1, cols=3, subplot_titles=("Box Plot", "Index vs Value", "Histogram"))

            plots = [
                go.Box(x=series, orientation="h", name="Box"),
                go.Scatter(x=df.index, y=series, mode="markers", name="Scatter"),
                go.Histogram(x=series, name="Histogram"),
            ]

            for i, trace in enumerate(plots, start=1):
                fig.add_trace(trace, row=1, col=i)

            fig.update_layout(title=f"Univariate Analysis: {col}", height=400, showlegend=False)

            figures[col] = fig

        return figures

    def plot_relationship(self, x: str, y: str):
        """
        Plots a visual representation of the relationship between two columns from the dataset,
        determining the most appropriate visualization type based on the data types of the
        specified columns. Supports scatter plots, box plots, and grouped bar charts.

        :param x: The name of the first column to analyze. It represents the x-axis in the plot.
        :type x: str
        :param y: The name of the second column to analyze. It represents the y-axis in the plot.
        :type y: str
        :return: A Plotly figure object visualizing the relationship between `x` and `y`. The type
                 of visualization changes based on the data types of the input columns. It can be
                 a scatter plot for numeric relations, a box plot for categorical-numeric relations,
                 or a grouped bar chart for categorical relationships.
        :rtype: plotly.graph_objs._figure.Figure or None
        :raises ValueError: If the column types are unsupported for generating a relationship plot.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        x_dtype = df[x].dtype
        y_dtype = df[y].dtype
        is_x_num = pd.api.types.is_numeric_dtype(x_dtype)
        is_y_num = pd.api.types.is_numeric_dtype(y_dtype)

        if is_x_num and is_y_num:
            return px.scatter(df, x=x, y=y, trendline="ols", title=f"Numeric Relationship: {x} vs {y}")
        if not is_x_num and is_y_num:
            return px.box(df, x=x, y=y, points="all", title=f"{x} vs {y}")
        if not is_x_num and not is_y_num:
            grouped = df.groupby([x, y]).size().reset_index(name="count")
            return px.bar(grouped, x=x, y="count", color=y, barmode="group", title=f"{x} vs {y}")
        else:
            raise ValueError("Unsupported column types")

    def plot_categorical_frequency(self, column: str):
        """
        Generates a bar chart visualizing the frequency distribution of a categorical column
        including absolute counts and percentages.

        :param column: Name of the categorical column to be analyzed.
        :type column: str
        :return: A Plotly figure representing the frequency distribution of the specified column.
        :rtype: plotly.graph_objs._figure.Figure or None
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        counts = df[column].value_counts()

        percentages = df[column].value_counts(normalize=True) * 100

        summary = pd.DataFrame({
            "count": counts,
            "percentage": percentages
        }).reset_index()

        summary.columns = [column, "count", "percentage"]

        fig = px.bar(summary, x=column, y="count", text=summary["percentage"].round(2).astype(str) + "%")
        fig.update_traces(textposition="outside")
        fig.update_layout(title=f"Frequency Distribution: {column}")

        return fig

    @staticmethod
    def _pearson(x, y):
        """
        Calculates the Pearson correlation coefficient between two pandas Series. The Pearson
        correlation measures the linear relationship between two datasets. It ranges from -1 to 1,
        where 1 represents a perfect positive linear relationship, -1 a perfect negative linear
        relationship, and 0 no linear relationship.

        :param x: A pandas Series representing the first dataset.
        :param y: A pandas Series representing the second dataset.
        :return: The Pearson correlation coefficient as a float. Returns NaN if there are fewer
            than two common elements between the datasets. Returns 0.0 if either dataset has a
            standard deviation of 0.
        """

        x = x.dropna()
        y = y.dropna()

        common = x.index.intersection(y.index)

        if len(common) < 2:
            return np.nan

        x, y = x.loc[common], y.loc[common]

        if x.std() == 0 or y.std() == 0:
            return 0.0

        return x.corr(y)

    @staticmethod
    def _cramers_v(x, y):
        """
        Computes Cramér's V statistic, a measure of association between two categorical variables,
        based on a chi-squared statistic. The function first replaces any missing values in the
        input data with a default value ("NA") to ensure valid calculations. It then computes
        a contingency table (confusion matrix) between the two variables, checks various
        conditions for validity, and calculates the Cramér's V.

        :param x: A pandas Series representing the first categorical variable. Missing values
            will be replaced with "NA".
        :param y: A pandas Series representing the second categorical variable. Missing values
            will be replaced with "NA".
        :return: A float representing the calculated Cramér's V value. Returns 0.0 if the
            contingency table is invalid or if any conditions for calculation are not met.
        """

        x = x.fillna("NA")
        y = y.fillna("NA")

        confusion_matrix = pd.crosstab(x, y)

        if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
            return 0.0

        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()

        if n == 0:
            return 0.0

        r, c = confusion_matrix.shape
        k = min(r - 1, c - 1)

        if k == 0:
            return 0.0

        return np.sqrt(chi2 / (n * k))

    @staticmethod
    def _eta_ratio(cat, num):
        """
        Calculate the eta ratio, a measure of association between a categorical variable
        and a numerical variable.

        This method computes the eta ratio by examining the variance between groups
        defined by the categorical variable compared to the total variance of the
        numerical variable. The formula uses the means of the numerical variable for
        groups defined by unique levels in the categorical variable.

        :param cat: A pandas Series representing the categorical variable.
        :param num: A pandas Series representing the numerical variable.
        :return: The computed eta ratio as a float. Returns NaN if there are less
            than two common index entries between the categorical and numerical
            variables, or if the total variance of the numerical variable is zero.
        :rtype: float
        """

        cat = cat.fillna("NA")
        num = num.dropna()

        common = cat.index.intersection(num.index)

        if len(common) < 2:
            return np.nan

        cat = cat.loc[common]
        num = num.loc[common]

        grand_mean = num.mean()

        ss_between = 0.0

        for level in cat.unique():
            group = num[cat == level]

            if len(group) == 0:
                continue

            ss_between += len(group) * (group.mean() - grand_mean) ** 2

        ss_total = ((num - grand_mean) ** 2).sum()

        if ss_total == 0:
            return 0.0

        return np.sqrt(ss_between / ss_total)

    def plot_all_associations_heatmap(self):
        """
        Generates a heatmap visualizing the associations between all numerical and categorical
        columns in the dataset. The associations are computed using Pearson correlation for
        numerical columns, Cramér's V for categorical columns, and eta ratio for mixed types.

        This method requires at least two columns in the dataset to perform the analysis.
        Numerical and categorical columns are identified internally from the dataset's
        data types.

        :raises ValueError: If fewer than two columns (numerical or categorical) are available
            for analysis.
        :return: A heatmap figure representing the associations between columns.
        :rtype: plotly.graph_objects.Figure
        """

        df = self.df.copy()

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        all_cols = num_cols + cat_cols

        if len(all_cols) < 2:
            raise ValueError("Not enough columns for association analysis.")

        matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

        series = {col: df[col] for col in all_cols}

        for i in all_cols:
            for j in all_cols:

                if i == j:
                    matrix.loc[i, j] = 1.0
                    continue

                xi = series[i]
                yj = series[j]

                if i in num_cols and j in num_cols:
                    val = self._pearson(xi, yj)

                elif i in cat_cols and j in cat_cols:
                    val = self._cramers_v(xi, yj)

                else:
                    if i in num_cols:
                        val = self._eta_ratio(yj, xi)
                    else:
                        val = self._eta_ratio(xi, yj)

                matrix.loc[i, j] = val

        fig = px.imshow(
            matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Unified Association Heatmap"
        )

        return fig

    def bar_chart(self, column: str, color: str | None = None, title: str | None = None) -> str | None:
        """
        Generates an HTML representation of a bar chart visualizing the frequency of
        values in the specified column of a dataframe. The chart can optionally group
        bars by color and display a custom title.

        :param column: The name of the column in the dataframe to visualize. Must be a
            valid column in the dataframe.
        :type column: str
        :param color: The name of the column used to colorize the bars for grouping.
            Defaults to None if no such grouping is needed.
        :type color: str | None
        :param title: The title of the bar chart. If not specified, a default title
            based on the column name will be used.
        :type title: str | None
        :return: An HTML string representation of the generated bar chart for embedding
            in web applications or other HTML-based outputs.
        :rtype: str
        :raises ValueError: If the specified column does not exist in the dataframe.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        if column not in df.columns:
            raise ValueError(f"{column} not in dataframe")

        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, "count"]
        fig = px.bar(value_counts, x=column, y="count", color=color, title=title or f"Bar Chart: {column}")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def pie_chart(self, column: str, title: str | None = None) -> str | None:
        """
        Generate an HTML representation of a pie chart using Plotly based on the distribution of values
        in a specified column within the dataframe. The method verifies if the dataframe is prepared
        for analysis and ensures that the specified column exists. If these conditions are not met,
        appropriate responses or exceptions are handled.

        :param column: The name of the column in the dataframe whose value distribution will be used
                       to generate the pie chart.
        :type column: str
        :param title: An optional parameter to specify the title for the pie chart.
                      Defaults to None, which uses a default chart title.
        :type title: str | None
        :return: A string containing the HTML representation of the pie chart, or None if the dataframe
                 is not prepared for analysis.
        :rtype: str | None
        :raises ValueError: If the specified column does not exist in the dataframe.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        if column not in df.columns:
            raise ValueError(f"{column} not in dataframe")

        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]

        fig = px.pie(counts, names=column, values="count", title=title or f"Pie Chart: {column}")

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def histogram(self, column: str, bins: int = 30, title: str | None = None) -> str | None:
        """
        Generates an HTML string representation of a histogram plot for the specified column in the DataFrame.

        The function creates a histogram using the provided column from the DataFrame, with the specified
        number of bins. An optional title can be specified for the plot. The histogram is generated using
        Plotly, and its HTML representation is returned. If the DataFrame is not ready or the specified
        column is not in the DataFrame, appropriate actions are taken.

        :param column: The name of the column in the DataFrame to plot.
        :param bins: The number of bins to use for the histogram. Defaults to 30.
        :param title: Optional title for the histogram plot. If not provided, a default title using the column
                      name will be used.
        :return: A string containing the HTML representation of the histogram plot, or None if the DataFrame
                 is not ready.
        :raises ValueError: If the specified column does not exist in the DataFrame.
        """

        if not self.df_ready():
            return None

        df = self.df.copy()

        if column not in df.columns:
            raise ValueError(f"{column} not in dataframe")

        fig = px.histogram(df, x=column, nbins=bins, title=title or f"Histogram: {column}")

        return fig.to_html(full_html=False, include_plotlyjs="cdn")
