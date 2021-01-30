import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 4.0)

print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss = 'warp') # choosing the LightFM model BPR/WARP etc.
model.fit(data['train'], epochs=30, num_threads=2) # fitting the data into the model in this case WARP model

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr() #preparing the data in to matrixand compressing sparse row format                                   
                          [user_id].indices]
        
        scores = model.predict(user_id, np.arange(n_items)) # predict scores between user item pairs or rating 
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:3]:
            print("        %s" % x)
        
        print("     Recommended:")
        
        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [3, 25, 451])