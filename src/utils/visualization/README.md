# Sankey Visualization Module

This module provides a `SankeyVisualizer` class and a `create_sankey_app` function to generate interactive, hierarchical Sankey diagrams for visualizing attention flow from machine learning models in genomics. It is designed to illustrate how attention is distributed from SNPs to genes and up through a biological ontology (like the Gene Ontology).

## Overview

The primary goal of this module is to make the complex, multi-layered attention data from a model interpretable. It does this by:

1.  **Processing Attention Data**: It takes a DataFrame of attention scores and recursively factorizes them, ensuring that the flow of attention is conserved as it moves up the hierarchy from SNPs to the root biological system.
2.  **Structuring the Hierarchy**: It uses an `SNPTreeParser` object to understand the relationships between SNPs, genes, and systems, and uses this structure to order and position the nodes in the diagram.
3.  **Generating Plots**: It uses Plotly to create a `go.Figure` object, providing a powerful and interactive visualization.
4.  **Creating an Interactive App**: It provides a factory function to quickly launch a Dash web application for interactively exploring the attention flow for different biological systems.

## Key Components

### `SankeyVisualizer` Class

This is the core class responsible for creating the Sankey diagrams. It is designed to be reusable and can be used to generate static plots (e.g., in a Jupyter notebook) or as the engine for the interactive Dash app.

-   **`__init__(tree_parser, ...)`**: Initializes the visualizer with an `SNPTreeParser` instance and optional annotation dictionaries for enriching the plot's tooltips.
-   **`plot(attention_df, ...)`**: The main public method. It takes the processed attention data and various configuration options and returns a `go.Figure` object. All the complex data preparation and layout calculations are handled by internal private methods.

### `create_sankey_app()` Function

This is a factory function that builds and returns a Dash application. It takes an initialized `SankeyVisualizer` instance and the attention data, and wires them up to interactive components like dropdowns and radio buttons. This separates the application logic from the plotting logic, making the code much cleaner and more modular.

## Workflow

1.  **Data Preparation**: The `_prepare_sankey_data` method is the main internal workhorse. It:
    *   Takes the raw attention data for a specific system.
    *   Calls `_factorize_attention_recursively` to correctly weight the attention flow through the hierarchy.
    *   Calls `_get_component_orders` to sort all the nodes (SNPs, genes, systems) in a deterministic way, which is crucial for a stable layout.
    *   Calls `_calculate_node_positions` to compute the `x` and `y` coordinates for each node, creating the distinct columns for SNPs, genes, and different levels of the system hierarchy.
    *   Calls `_get_sankey_link_values` to prepare the data for the links (the flows) between the nodes.

2.  **Plot Generation**: The `plot` method takes the highly structured data from `_prepare_sankey_data` and passes it directly to `plotly.graph_objects.Sankey` to generate the figure.

3.  **App Creation**: The `create_sankey_app` function defines the Dash layout (e.g., dropdowns, graph component) and the callback function. The callback function is responsible for responding to user input (like selecting a new system), calling the `visualizer.plot` method with the appropriate parameters, and updating the graph.

## Usage Example

### Generating a Static Plot

```python
from src.utils.tree import SNPTreeParser
from src.utils.visualization.sankey import SankeyVisualizer
import pandas as pd

# 1. Initialize the SNPTreeParser
tree_parser = SNPTreeParser(...)

# 2. Load and process your attention data into a DataFrame
# This is a simplified example; your actual data will be more complex
attention_data = pd.read_csv('path/to/attention.csv', index_col=[0, 1, 2])

# 3. Initialize the visualizer
visualizer = SankeyVisualizer(tree_parser)

# 4. Generate the plot
fig = visualizer.plot(
    attention_df=attention_data,
    target_go='GO:0008150',
    direction='forward',
    genotypes=('homozygous', 'heterozygous'),
    title="My First Sankey Plot"
)

# 5. Show the plot
fig.show()
```

### Launching the Interactive Dash App

```python
# (Continuing from the previous example)
from src.utils.visualization.sankey import create_sankey_app

# A list of GO terms you want to be able to select in the app
go_term_list = ['GO:0008150', 'GO:0006915', 'GO:0007049']

# Create the Dash app instance
app = create_sankey_app(
    visualizer=visualizer,
    attention_df=attention_data, # The full attention DataFrame
    target_go_list=go_term_list,
    initial_go='GO:0008150'
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```
