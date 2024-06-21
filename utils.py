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

def sample_best_perform_user_item_interactions(user_embeddings_pool, user_ids_pool, item_embeddings_pool, item_ids_pool, ratings, num_users=5, n=10):
    """Sample the best performed user embeddings and their nearest rated item embeddings according to the precision@n

    Parameters
    ----------
    user_embeddings_pool : np.array
        The pool of user embeddings.
    user_ids_pool : np.array
        The pool of user ids which correspond to the user embeddings.
    item_embeddings_pool : np.array
        The pool of item embeddings.
    item_ids_pool : np.array
        The pool of item ids which correspond to the item embeddings.
    ratings : pd.DataFrame
        The ratings DataFrame from `load_ratings()` function.
    num_users : int
        The number of users to sample.
    n : int
        The number of nearest neighbors to sample for each user.

    Returns
    -------
    np.array
        The sampled user embeddings.
    np.array
        The sampled item embeddings.
    float
        The global precision@n.
    float
        The sampled precision@n.
    """

    # calculate the predicted ratings
    predicted_ratings = np.dot(user_embeddings_pool.astype(float), item_embeddings_pool.T.astype(float))

    # get the nearest n neighbors for each center embedding
    nearest_n_item_indices = np.argsort(predicted_ratings, axis=1)[:, -n:]
    nearest_n_items_id = item_ids_pool[nearest_n_item_indices]

    # calculate the precision@n with ratings as the ground truth
    precision_at_n = []
    for i in range(len(user_embeddings_pool)):
        # user_embedding = user_embeddings_pool[i]
        user_id = user_ids_pool[i]
        user_ratings = ratings[ratings['user_id'] == user_id]
        top_n_neighbors_for_user = nearest_n_items_id[i]
        precision_at_n_for_user = len(user_ratings[user_ratings['item_id'].isin(top_n_neighbors_for_user)]) / n
        precision_at_n.append(precision_at_n_for_user)

    # pick top N user indices with highest precision@n
    top_user_indices = np.argsort(precision_at_n)[-num_users:]
    global_precision_at_n = np.mean(precision_at_n)
    sampled_precision_at_n = np.mean(np.array(precision_at_n)[top_user_indices])

    # sample the nearest rated item embeddings for each top user
    top_item_embeddings = []
    for user_index in top_user_indices:
        user_embedding = user_embeddings_pool[user_index]
        user_id = user_ids_pool[user_index]
        user_ratings = ratings[ratings['user_id'] == user_id]
        user_rated_item_ids = user_ratings['item_id'].values
        user_rated_item_indices = np.where(np.isin(item_ids_pool, user_rated_item_ids))[0]
        user_rated_item_embeddings = item_embeddings_pool[user_rated_item_indices]
        _, sampled_neighbor_embeddings = sample_nearest_neighbors(np.array([user_embedding]), 
                                                                                           user_rated_item_embeddings, 
                                                                                           sample_size=1, 
                                                                                           neighbor_size=n)
        top_item_embeddings.append(sampled_neighbor_embeddings)

    user_embeddings = user_embeddings_pool[top_user_indices]
    top_item_embeddings = np.array(top_item_embeddings).reshape(-1, top_item_embeddings[0].shape[-1])
    return user_embeddings, top_item_embeddings, global_precision_at_n, sampled_precision_at_n

def plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='tsne', n_iter=10000, perplexity=10):
    """Visualize the embeddings using t-SNE or PCA.
    
    Parameters
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

    Returns
    -------
    None
    """

    # concat item and user embeddings
    X = np.concatenate((sampled_center_embeddings, sampled_neighbor_embeddings), axis=0)

    if method == 'tsne':
        # t-SNE
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random', n_iter=10000, perplexity=10, verbose=False).fit_transform(X)
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
        plt.scatter(sampled_center_embeddings_tsne[i, 0], sampled_center_embeddings_tsne[i, 1], label='center', s=30, c=[[r, g, b]])
        for j in range(num_neighbor_samples):
            # plot neighbor embeddings with color based on hue and 100% brightness, 50% saturation
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.7)
            plt.scatter(sampled_neighbor_embeddings_tsne[i*num_neighbor_samples+j, 0], sampled_neighbor_embeddings_tsne[i*num_neighbor_samples+j, 1], label='neighbor', s=10, c=[[r, g, b]])
        hue += hue_step

    plt.show()