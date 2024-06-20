#%%
# Import the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys

import numpy as np
random_seed = np.random.randint(0, 1000) # 995 looks good. Fix it instead of randomizing it for reproducibility.
print("Random seed: ", random_seed)
np.random.seed(random_seed)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

#%%
# Load rating.dat

ratings = pd.read_csv('data/ratings.dat', sep='::', header=None)
ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
print("Ratings shape: ", ratings.shape)
print("Unique users: ", ratings['user_id'].nunique())
print("Unique items: ", ratings['item_id'].nunique())
print()

#%%
# Load the user and item embeddings

user_embeddings_df = pd.read_csv('data/user_embeddings.parquet', sep='\t', header=None)
user_embeddings_df.columns = ['user_id', 'embeddings']
user_embeddings_df['embeddings'] = user_embeddings_df['embeddings'].str.split(' ')

user_ids = user_embeddings_df['user_id'].values
user_embeddings = np.array(user_embeddings_df['embeddings'].values.tolist(), dtype=float)
print("User embeddings shape: ", user_embeddings.shape)

item_embeddings_df = pd.read_csv('data/item_embeddings.parquet', sep='\t', header=None)
item_embeddings_df.columns = ['item_id', 'embeddings']
item_embeddings_df['embeddings'] = item_embeddings_df['embeddings'].str.split(' ')

item_ids = item_embeddings_df['item_id'].values
item_embeddings = np.array(item_embeddings_df['embeddings'].values.tolist(), dtype=float)
print("Item embeddings shape: ", item_embeddings.shape)

#%%
# Generate pseudo un-trained embeddings using Xavier initialization.

def xavier_init(shape):
    return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))

# random initialization also provided for comparison. although it is not used in the original LightGCN design, you can still try it to see why Xavier is generally preferred.
def random_init(shape):
    return np.random.randn(*shape)

user_embeddings_init = xavier_init((user_embeddings.shape[0], user_embeddings.shape[1]))
# user_embeddings_init = random_init((user_embeddings.shape[0], user_embeddings.shape[1]))
print("User embeddings pseudo init shape: ", user_embeddings_init.shape)

item_embeddings_init = xavier_init((item_embeddings.shape[0], item_embeddings.shape[1]))
# item_embeddings_init = random_init((item_embeddings.shape[0], item_embeddings.shape[1]))
print("Item embeddings pseudo init shape: ", item_embeddings_init.shape)

#%%
# Declare functions: Sample N user embeddings with highest precision@n and their nearest rated item embeddings

# sample nearest neighbors for each center embedding
def sample_nearest_embeddings(center_embeddings_pool, neighbor_embeddings_pool, sample_size=3, neighbor_size=10):
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

# sample top user embeddings with highest precision@n and their nearest rated item embeddings
def sample_top_interactions_embeddings(user_embeddings_pool, user_ids_pool, item_embeddings_pool, item_ids_pool, ratings, num_users=5, n=10):
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
    average_precision_at_n = np.mean(precision_at_n)

    # sample the nearest rated item embeddings for each top user
    top_item_embeddings = []
    for user_index in top_user_indices:
        user_embedding = user_embeddings_pool[user_index]
        user_id = user_ids_pool[user_index]
        user_ratings = ratings[ratings['user_id'] == user_id]
        user_rated_item_ids = user_ratings['item_id'].values
        user_rated_item_indices = np.where(np.isin(item_ids_pool, user_rated_item_ids))[0]
        user_rated_item_embeddings = item_embeddings_pool[user_rated_item_indices]
        sampled_center_embeddings, sampled_neighbor_embeddings = sample_nearest_embeddings(np.array([user_embedding]), 
                                                                                           user_rated_item_embeddings, 
                                                                                           sample_size=1, 
                                                                                           neighbor_size=n)
        top_item_embeddings.append(sampled_neighbor_embeddings)

    user_embeddings = user_embeddings_pool[top_user_indices]
    top_item_embeddings = np.array(top_item_embeddings).reshape(-1, top_item_embeddings[0].shape[-1])
    return user_embeddings, top_item_embeddings, average_precision_at_n

#%%
# Declare function: Visualize embeddings using t-SNE

# visualize embeddings using t-SNE or PCA
def plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='tsne', n_iter=10000, perplexity=10):
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

#%%
# Sample and visualize the embeddings with pseudo un-trained embeddings and trained embeddings

num_center_samples = 5 # number of user embeddings to sample
num_neighbor_samples = 20  # number of nearest rated item embeddings to sample

# visualize pseudo un-trained embeddings
print("Visualizing pseudo un-trained embeddings...")
sampled_center_embeddings, sampled_neighbor_embeddings, average_precision_at_n = sample_top_interactions_embeddings(user_embeddings_init, 
                                                                                                                    user_ids, 
                                                                                                                    item_embeddings_init, 
                                                                                                                    item_ids, 
                                                                                                                    ratings, 
                                                                                                                    num_users=num_center_samples, 
                                                                                                                    n=num_neighbor_samples)
print(f"Average precision@{num_neighbor_samples} for pseudo un-trained embeddings: {average_precision_at_n}")

# visualize with T-SNE
plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='tsne')

# visualize with PCA
# plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='pca')

#%%
# visualize trained embeddings
print("Visualizing trained embeddings...")
sampled_center_embeddings, sampled_neighbor_embeddings, average_precision_at_n = sample_top_interactions_embeddings(user_embeddings, 
                                                                                                                    user_ids, 
                                                                                                                    item_embeddings, 
                                                                                                                    item_ids, 
                                                                                                                    ratings, 
                                                                                                                    num_users=num_center_samples, 
                                                                                                                    n=num_neighbor_samples)
print(f"Average precision@{num_neighbor_samples} for trained embeddings: {average_precision_at_n}")

# visualize with T-SNE
plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='tsne')

# visualize with PCA
# plot_embeddings(sampled_center_embeddings, sampled_neighbor_embeddings, num_center_samples, num_neighbor_samples, method='pca')

#%%