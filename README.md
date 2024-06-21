# embeddings-visualization-for-recsys

## Quick Start
Run `visualize.py`. 

[VSCode Python Interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) is recommended.

Expected executed results should be like `executed_result_visualize_pca.html` or `executed_result_visualize_tsne.html` which could be opened in a browser.

## Features

### 1. Visualize Embeddings
Visualize embeddings in 2D space using PCA or t-SNE. Please see `# Sample and visualize the embeddings with pseudo un-trained embeddings` and `# Sample and visualize the embeddings with trained embeddings` section in `visualize.py` for more details.

### 2. Comparison of Embeddings Before and After Training
The `# Generate pseudo un-trained embeddings using Xavier initialization.` section provides a way to generate **pseudo un-trained embeddings** using *Xavier initialization* or *Random initialization*. By using the generated pseudo un-trained embeddings, you can compare the embeddings before and after training.

In original LightGCN implementation, the embeddings are initialized using *Xavier initialization*. However, you can also use *Random initialization* and see the difference. It will be helpful to understand why *Xavier initialization* is preferred in most cases.
