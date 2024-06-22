import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_ratings(filename) -> pd.DataFrame:
    """Load the ratings.dat file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        The path to the ratings.dat file.
        The file should be formatted as `user_id::item_id::rating::timestamp`.

    Returns
    -------
    pd.DataFrame
        The ratings DataFrame.
    """

    ratings = pd.read_csv(filename, sep='::', header=None)
    ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    return ratings

def load_embeddings(filename):
    """Load the embeddings file into numpy arrays.

    Parameters
    ----------
    filename : str
        The path to the embeddings file.
        The file should be formatted as a **tab-separated** file (ie. tsv) with the first column as the id and the second column as the embeddings.
        The embeddings should be a long string of **space-separated** values.

    Returns
    -------
    np.array
        The ids.
    np.array
        The embeddings.
    """

    embeddings_df = pd.read_csv(filename, sep='\t', header=None)
    embeddings_df.columns = ['id', 'embeddings']
    embeddings_df['embeddings'] = embeddings_df['embeddings'].str.split(' ')
    ids = embeddings_df['id'].values
    embeddings = np.array(embeddings_df['embeddings'].values.tolist(), dtype=float)
    return ids, embeddings

def xavier_init(shape):
    """Xavier initialization for the embeddings.

    Parameters
    ----------
    shape : tuple
        The shape of the embeddings. (num_embeddings, embedding_dim)

    Returns
    -------
    np.array
        The initialized embeddings.
    """

    return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))

def random_init(shape):
    """Random initialization for the embeddings.

    Parameters
    ----------
    shape : tuple
        The shape of the embeddings. (num_embeddings, embedding_dim)

    Returns
    -------
    np.array
        The initialized embeddings.
    """

    return np.random.randn(*shape)

def sample_nearest_neighbors(center_embeddings_pool, neighbor_embeddings_pool, sample_size=3, neighbor_size=10):
    """Randomly samples `sample_size` center embeddings and their `neighbor_size` nearest neighbors from the pool of center and neighbor embeddings.

    Parameters
    ----------
    center_embeddings_pool : np.array
        The pool of center embeddings.
    neighbor_embeddings_pool : np.array
        The pool of neighbor embeddings.
    sample_size : int
        The number of center embeddings to sample.
    neighbor_size : int
        The number of nearest neighbors to sample for each center embedding.

    Returns
    -------
    np.array
        The sampled center embeddings.
    np.array
        The sampled neighbor embeddings.
    """

    center_embeddings_pool = center_embeddings_pool.copy()
    neighbor_embeddings_pool = neighbor_embeddings_pool.copy()
    sampled_center_embeddings = center_embeddings_pool[np.random.choice(center_embeddings_pool.shape[0], sample_size, replace=False)]
    sampled_neighbor_embeddings = []
    for center_embedding in sampled_center_embeddings:
        center_embedding = np.array(center_embedding).astype(float)
        distances = np.linalg.norm(neighbor_embeddings_pool.astype(float) - center_embedding, axis=1)
        sampled_neighbor_indices = np.argsort(distances)[:neighbor_size]
        sampled_neighbor_embeddings.append(neighbor_embeddings_pool[sampled_neighbor_indices])
    sampled_neighbor_embeddings = np.array(sampled_neighbor_embeddings)
    
    # flatten the sampled_neighbor_embeddings
    sampled_neighbor_embeddings = sampled_neighbor_embeddings.reshape(-1, sampled_neighbor_embeddings.shape[-1])
    return sampled_center_embeddings, sampled_neighbor_embeddings

def sample_best_perform_user_item_interactions(center_embeddings_pool, center_ids_pool, neighbor_embeddings_pool, neighbor_ids_pool, ratings, c=5, n=10):
    """Sample the best performed user embeddings and their nearest rated item embeddings according to the precision@n

    Parameters
    ----------
    center_embeddings_pool : np.array
        The pool of center node embeddings.
    center_ids_pool : np.array
        The pool of center node ids which correspond to the center node embeddings.
    neighbor_embeddings_pool : np.array
        The pool of neighbor node embeddings.
    neighbor_ids_pool : np.array
        The pool of neighbor node ids which correspond to the neighbor node embeddings.
    ratings : pd.DataFrame
        The ratings DataFrame contain two columns: 'center_id' and 'neighbor_id'.
    c : int
        The number of center nodes to sample.
    n : int
        The number of nearest neighbors to sample for each center node.

    Returns
    -------
    np.array
        The sampled center node embeddings.
    np.array
        The sampled neighbor node embeddings.
    float
        The global precision@n.
    float
        The sampled precision@n.
    """

    # calculate the predicted ratings
    predicted_ratings = np.dot(center_embeddings_pool.astype(float), neighbor_embeddings_pool.T.astype(float))

    # get the nearest n neighbors for each center embedding
    nearest_n_neighbor_indices = np.argsort(predicted_ratings, axis=1)[:, -n:]
    nearest_n_neighbor_ids = neighbor_ids_pool[nearest_n_neighbor_indices]

    # calculate the precision@n with ratings as the ground truth
    precision_at_n = []
    for i in range(len(center_embeddings_pool)):
        # user_embedding = user_embeddings_pool[i]
        center_id = center_ids_pool[i]
        rating = ratings[ratings['center_id'] == center_id]
        nearest_n_neighbor_ids_for_the_center = nearest_n_neighbor_ids[i]
        precision_at_n_for_the_center = len(rating[rating['neighbor_id'].isin(nearest_n_neighbor_ids_for_the_center)]) / n
        precision_at_n.append(precision_at_n_for_the_center)

    # pick top N user indices with highest precision@n
    best_performed_center_indices = np.argsort(precision_at_n)[-c:]
    global_precision_at_n = np.mean(precision_at_n)
    sampled_precision_at_n = np.mean(np.array(precision_at_n)[best_performed_center_indices])

    # sample the nearest rated item embeddings for each top user
    sampled_neighbor_embeddings = []
    for center_index in best_performed_center_indices:
        center_embedding = center_embeddings_pool[center_index]
        center_id = center_ids_pool[center_index]
        rating = ratings[ratings['center_id'] == center_id]
        center_interacted_neighbor_ids = rating['neighbor_id'].values
        center_interacted_neighbor_indices = np.where(np.isin(neighbor_ids_pool, center_interacted_neighbor_ids))[0]
        center_interacted_neighbor_embeddings = neighbor_embeddings_pool[center_interacted_neighbor_indices]
        _, sampled_neighbor_embeddings_for_the_center = sample_nearest_neighbors(np.array([center_embedding]), 
                                                                                           center_interacted_neighbor_embeddings, 
                                                                                           sample_size=1, 
                                                                                           neighbor_size=n)
        sampled_neighbor_embeddings.append(sampled_neighbor_embeddings_for_the_center)

    center_embeddings = center_embeddings_pool[best_performed_center_indices]
    sampled_neighbor_embeddings = np.array(sampled_neighbor_embeddings).reshape(-1, sampled_neighbor_embeddings[0].shape[-1])
    return center_embeddings, sampled_neighbor_embeddings, global_precision_at_n, sampled_precision_at_n

def plot_embeddings_heterogeneous(sampled_center_embeddings, sampled_neighbor_embeddings, method='tsne', n_iter=10000, perplexity=10, center_dot_size=30, neighbor_dot_size=10):
    """Visualize the embeddings using t-SNE or PCA.
    
    Parameters`
    ----------
    sampled_center_embeddings : np.array
        The sampled center embeddings.
    sampled_neighbor_embeddings : np.array
        The sampled neighbor embeddings.
    num_center_samples : int
        The number of center embeddings sampled.
    num_neighbor_samples : int
        The number of neighbor embeddings sampled.
    method : str
        The method to use for visualization. Choose either 'tsne' or 'pca'.
    n_iter : int
        The number of iterations for t-SNE. Default is 10000.
    perplexity : int
        The perplexity for t-SNE. Default is 10. This parameter is related to the number of `num_neighbor_samples`, consider changing it if the visualization is not clear.
    center_dot_size : int
        The size of the center dots in the plot.
    neighbor_dot_size : int
        The size of the neighbor dots in the plot.

    Returns
    -------
    None
    """

    # concat item and user embeddings
    num_center_samples = sampled_center_embeddings.shape[0]
    num_neighbor_samples = sampled_neighbor_embeddings.shape[0] // num_center_samples
    X = np.concatenate((sampled_center_embeddings, sampled_neighbor_embeddings), axis=0)

    if method == 'tsne':
        # t-SNE
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', n_iter=n_iter, perplexity=perplexity, verbose=False).fit_transform(X)
    elif method == 'pca':
        # PCA
        X_embedded = PCA(n_components=2).fit_transform(X)
    else:
        raise ValueError("Invalid method. Choose either 'tsne' or 'pca'.")
    
    sampled_center_embeddings_tsne = X_embedded[:num_center_samples]
    sampled_neighbor_embeddings_tsne = X_embedded[num_center_samples:]

    # plot the embeddings
    plt.figure(figsize=(5, 5))
    hue = 0
    hue_step = 1 / num_center_samples
    for i in range(num_center_samples):
        # plot center embeddings with color based on hue and 100% brightness, 100% saturation
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 0.7)
        plt.scatter(sampled_center_embeddings_tsne[i, 0], sampled_center_embeddings_tsne[i, 1], label='center', s=center_dot_size, c=[[r, g, b]])
        for j in range(num_neighbor_samples):
            # plot neighbor embeddings with color based on hue and 100% brightness, 50% saturation
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.7)
            plt.scatter(sampled_neighbor_embeddings_tsne[i*num_neighbor_samples+j, 0], sampled_neighbor_embeddings_tsne[i*num_neighbor_samples+j, 1], label='neighbor', s=neighbor_dot_size, c=[[r, g, b]])
        hue += hue_step

    plt.show()

def plot_embeddings_homogeneous(sampled_embeddings, method='tsne', n_iter=10000, perplexity=10, dot_size=10):
    """Visualize the embeddings using t-SNE or PCA.
    
    Parameters`
    ----------
    sampled_embeddings : np.array
        The sampled embeddings.
    num_samples : int
        The number of embeddings sampled.
    method : str
        The method to use for visualization. Choose either 'tsne' or 'pca'.
    n_iter : int
        The number of iterations for t-SNE. Default is 10000.
    perplexity : int
        The perplexity for t-SNE. Default is 10. This parameter is related to the number of `num_samples`, consider changing it if the visualization is not clear.
    dot_size : int
        The size of the dots in the plot.

    Returns
    -------
    None
    """

    if method == 'tsne':
        # t-SNE
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', n_iter=n_iter, perplexity=perplexity, verbose=False).fit_transform(sampled_embeddings)
    elif method == 'pca':
        # PCA
        X_embedded = PCA(n_components=2).fit_transform(sampled_embeddings)
    else:
        raise ValueError("Invalid method. Choose either 'tsne' or 'pca'.")
    
    # plot the embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=dot_size)
    plt.show()