from surprise import accuracy, Dataset, SVDpp, SVD
import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')
dataset = pd.DataFrame(data.raw_ratings, columns = ['uid', 'iid', 'rating', 'timestamp'])

trainset, testset = train_test_split(data, test_size=0.2)

pp = SVDpp()
pp.fit(trainset)
test_pred = pp.test(testset)
print(accuracy.rmse(test_pred, verbose=True))

svd = SVD()
svd.fit(trainset)    
test_pred = svd.test(testset)
print(accuracy.rmse(test_pred, verbose=True))


def generate_recommendation(uid, model, dataset, thresh=4.0, amount=5):
    all_titles = list(dataset['iid'].values)
    users_seen_titles = dataset[dataset['uid'] == uid]['iid']
    titles = np.array(list(set(all_titles) - set(users_seen_titles)))

    np.random.shuffle(titles)
    
    rec_list = []
    for title in titles:
        review_prediction = model.predict(uid=uid, iid=title)
       
        rating = review_prediction.est

        if rating >= thresh:
            rec_list.append((title, round(rating, 2)))
            
            if len(rec_list) >= amount:
                return rec_list
            
            
print(generate_recommendation(294, svd, dataset))         
print(generate_recommendation(294, pp, dataset))   
 








print()

#Blending sizing





