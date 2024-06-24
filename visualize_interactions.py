#%%
# Import the necessary libraries

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
random_seed = np.random.randint(0, 1000) # 995 looks good. Fix it instead of randomizing it for reproducibility.
print("Random seed: ", random_seed)
np.random.seed(random_seed)

from utils import *

#%%
# Load rating.dat

ratings = load_ratings('data/ratings.dat')
print("Ratings shape: ", ratings.shape)
print("Unique users: ", ratings['user_id'].nunique())
print("Unique items: ", ratings['item_id'].nunique())
print()

# rename columns
# ratings.columns = ['center_id', 'neighbor_id', 'rating', 'timestamp'] # create user centered visualization.
ratings.columns = ['neighbor_id', 'center_id', 'rating', 'timestamp'] # create item centered visualization.

#%%
# Load the user and item embeddings

user_ids, user_embeddings = load_embeddings('data/user_embeddings.parquet')
print("User embeddings shape: ", user_embeddings.shape)

item_ids, item_embeddings = load_embeddings('data/item_embeddings.parquet')
print("Item embeddings shape: ", item_embeddings.shape)

#%%
# Generate pseudo un-trained embeddings using Xavier initialization.

user_embeddings_init = xavier_init((user_embeddings.shape[0], user_embeddings.shape[1]))
# user_embeddings_init = random_init((user_embeddings.shape[0], user_embeddings.shape[1]))
print("User embeddings pseudo init shape: ", user_embeddings_init.shape)

item_embeddings_init = xavier_init((item_embeddings.shape[0], item_embeddings.shape[1]))
# item_embeddings_init = random_init((item_embeddings.shape[0], item_embeddings.shape[1]))
print("Item embeddings pseudo init shape: ", item_embeddings_init.shape)

#%%
# Sample and visualize the embeddings with pseudo un-trained embeddings.

num_center_samples = 5 # number of user embeddings to sample
num_neighbor_samples = 20  # number of nearest rated item embeddings to sample
n=20 # precision@n

# visualize pseudo un-trained embeddings
print("Visualizing pseudo un-trained embeddings...")

# # user centered visualization
# sampled_center_embeddings, sampled_neighbor_embeddings, global_precision_at_n, sampled_precision_at_n = sample_best_perform_user_item_interactions(user_embeddings_init, 
#                                                                                                                     user_ids, 
#                                                                                                                     item_embeddings_init, 
#                                                                                                                     item_ids, 
#                                                                                                                     ratings_sampling=ratings,
#                                                                                                                     ratings_validation=ratings,
#                                                                                                                     num_centers=num_center_samples,
#                                                                                                                     num_neighbors=num_neighbor_samples,
#                                                                                                                     n=n)

# item centered visualization
sampled_center_embeddings, sampled_neighbor_embeddings, global_precision_at_n, sampled_precision_at_n = sample_best_perform_user_item_interactions(item_embeddings_init,
                                                                                                                    item_ids,
                                                                                                                    user_embeddings_init,
                                                                                                                    user_ids,
                                                                                                                    ratings_sampling=ratings,
                                                                                                                    ratings_validation=ratings,
                                                                                                                    num_centers=num_center_samples,
                                                                                                                    num_neighbors=num_neighbor_samples,
                                                                                                                    n=n)

print(f"Global precision@{num_neighbor_samples} for pseudo un-trained embeddings: {global_precision_at_n:0.4f}")
print(f"Sampled precision@{num_neighbor_samples} for pseudo un-trained embeddings: {sampled_precision_at_n:0.4f}")

# visualize with T-SNE
plot_embeddings_heterogeneous(sampled_center_embeddings, sampled_neighbor_embeddings, method='tsne')

# visualize with PCA
# plot_embeddings_heterogeneous(sampled_center_embeddings, sampled_neighbor_embeddings, method='pca')

#%%
# Sample and visualize the embeddings with trained embeddings.

# visualize trained embeddings
print("Visualizing trained embeddings...")

# user centered visualization
# sampled_center_embeddings, sampled_neighbor_embeddings, global_precision_at_n, sampled_precision_at_n = sample_best_perform_user_item_interactions(user_embeddings, 
#                                                                                                                     user_ids, 
#                                                                                                                     item_embeddings, 
#                                                                                                                     item_ids, 
#                                                                                                                     ratings_sampling=ratings,
#                                                                                                                     ratings_validation=ratings,
#                                                                                                                     num_centers=num_center_samples,
#                                                                                                                     num_neighbors=num_neighbor_samples,
#                                                                                                                     n=n)

# item centered visualization
sampled_center_embeddings, sampled_neighbor_embeddings, global_precision_at_n, sampled_precision_at_n = sample_best_perform_user_item_interactions(item_embeddings,
                                                                                                                    item_ids,   
                                                                                                                    user_embeddings,
                                                                                                                    user_ids,
                                                                                                                    ratings_sampling=ratings,
                                                                                                                    ratings_validation=ratings,
                                                                                                                    num_centers=num_center_samples,
                                                                                                                    num_neighbors=num_neighbor_samples,
                                                                                                                    n=n)

print(f"Global precision@{num_neighbor_samples} for trained embeddings: {global_precision_at_n:0.4f}")
print(f"Sampled precision@{num_neighbor_samples} for trained embeddings: {sampled_precision_at_n:0.4f}")

# visualize with T-SNE
plot_embeddings_heterogeneous(sampled_center_embeddings, sampled_neighbor_embeddings, method='tsne')

# visualize with PCA
# plot_embeddings_heterogeneous(sampled_center_embeddings, sampled_neighbor_embeddings, method='pca')

#%%