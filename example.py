import numpy as np

from lightfm.datasets import fetch_movielens

movielens = fetch_movielens()

for key, value in movielens.items():
    print(key, type(value), value.shape)
    
train = movielens['train']
test = movielens['test']



from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM()
model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10, train_interactions=train).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))



model = LightFM(learning_rate=0.05, loss='warp')

model.fit_partial(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10, train_interactions=train).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))


csr = movielens['train'].tocsr()
three = csr[3]
ind = three.indices
                                   

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()                                    
                          [user_id].indices]
        
        scores = model.predict(user_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:3]:
            print("        %s" % x)
        
        print("     Recommended:")
        
        for x in top_items[:3]:
            print("        %s" % x)
            
            
            
    
#sample_recommendation(model, movielens, [3, 25, 450])        
            
