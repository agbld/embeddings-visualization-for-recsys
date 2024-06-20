# embeddings-visualization-for-recsys

## Quick Start
Run `visualize.py`. 

[VSCode Python Interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) is recommended.

Expected executed results should be like `executed_result_visualize_pca.html` or `executed_result_visualize_tsne.html` which could be opened in a browser.

## Features

### 1. Visualize Embeddings
Visualize embeddings in 2D space using PCA or t-SNE. Please see `# Declare function: Visualize embeddings using t-SNE` section in `visualize.py` for more details.
* For `T-SNE`: Uncomment *185*, *203* lines and comment *186*, *204* lines in `visualize.py` to use T-SNE.
* For `PCA`: Uncomment *186*, *204* lines and comment *185*, *203* lines in `visualize.py` to use PCA.

### 2. Comparison of Embeddings Before and After Training
The `# Generate pseudo un-trained embeddings using Xavier initialization.` section provides a way to generate **pseudo un-trained embeddings** using *Xavier initialization* or *Random initialization*. By using the generated pseudo un-trained embeddings, you can compare the embeddings before and after training.