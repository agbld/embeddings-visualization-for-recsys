#%%
# Import the necessary libraries

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
random_seed = np.random.randint(0, 1000) # 995 looks good. Fix it instead of randomizing it for reproducibility.
print("Random seed: ", random_seed)
np.random.seed(random_seed)

from utils import load_embeddings, xavier_init, random_init, plot_embeddings_homogeneous

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

num_samles = 2000 # number of node embeddings to sample

# sampled_node_embeddings = user_embeddings_init[np.random.choice(user_embeddings_init.shape[0], num_samles, replace=False)] # sample user embeddings
sampled_node_embeddings = item_embeddings_init[np.random.choice(item_embeddings_init.shape[0], num_samles, replace=False)] # sample item embeddings

# visualize pseudo un-trained embeddings
print("Visualizing pseudo un-trained embeddings...")
plot_embeddings_homogeneous(sampled_node_embeddings, method='tsne', n_iter=10000, perplexity=50, dot_size=2)

#%%
# Sample and visualize the embeddings with trained embeddings.

sampled_node_embeddings = user_embeddings[np.random.choice(user_embeddings.shape[0], num_samles, replace=False)] # sample user embeddings
# sampled_node_embeddings = item_embeddings[np.random.choice(item_embeddings.shape[0], num_samles, replace=False)] # sample item embeddings

# visualize trained embeddings
print("Visualizing trained embeddings...")
plot_embeddings_homogeneous(sampled_node_embeddings, method='tsne', n_iter=10000, perplexity=50, dot_size=2)

#%%