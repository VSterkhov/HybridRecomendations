import numpy as np
import pandas as pd
from tqdm import tqdm
from lightfm.datasets import fetch_movielens
from scipy.spatial.distance import cityblock, cosine, euclidean, hamming, jaccard



movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

movies_with_ratings = movies.merge(ratings, on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

num_users = movies_with_ratings.userId.unique().shape[0]

movie_vector = {}

for movie, group in tqdm(movies_with_ratings.groupby('title')):
    movie_vector[movie] = np.zeros(num_users)
    
    for i in range(len(group.userId.values)):
        u = group.userId.values[i]
        r = group.rating.values[i]
        movie_vector[movie][int(u - 1)] = r
        
        
        
print(movie_vector['Toy Story (1995)'])


my_fav_film = 'Fight Club (1999)'

titles = []
distances = []

for key in tqdm(movie_vector.keys()):
    if key == my_fav_film:
        continue
    
    titles.append(key)
    distances.append(cosine(movie_vector[my_fav_film], movie_vector[key]))

len(distances)

best_indexes = np.argsort(distances)[:10]
best_indexes

    
best_movies = [(titles[i], distances[i]) for i in best_indexes]

for m in best_movies:
    print(m)
  
    
  
    
from surprise import KNNWithMeans, SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import pandas as pd



dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
})


reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset, reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=1)
  
algo = KNNWithMeans(k=50, sim_options={
    'name': 'cosine',
    'user_based': True  # compute  similarities between users
})
algo.fit(trainset)    
  
algo2 = KNNWithMeans(k=50, sim_options={
    'name': 'cosine',
    'user_based': False  # compute similarities between items
})
algo2.fit(trainset)

test_pred = algo.test(testset)
print(accuracy.rmse(test_pred, verbose=True))


print(algo.predict(uid=3, iid='Fight Club (1999)'))


def generate_recommendation(uid, model, dataset, thresh=4.5, amount=5):
    all_titles = list(dataset['iid'].values)
    users_seen_titles = dataset[dataset['uid'] == uid]['iid']
    titles = np.array(list(set(all_titles) - set(users_seen_titles)))

    np.random.shuffle(titles)
    
    rec_list = []
    for title in titles:
        review_prediction = model.predict(uid=uid, iid=title)
        print(review_prediction)
        rating = review_prediction.est

        if rating >= thresh:
            rec_list.append((title, round(rating, 2)))
            
            if len(rec_list) >= amount:
                return rec_list
            
            
#print(generate_recommendation(3, algo, dataset))         

print(generate_recommendation(3, algo2, dataset))  

#print(algo.predict(uid=3, iid='Mortal Kombat (1995)').est)



'''

from lightfm import LightFM

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

from lightfm.evaluation import precision_at_k

print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())
'''

