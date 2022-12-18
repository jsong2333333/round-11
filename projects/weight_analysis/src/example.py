import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def bootstrap_performance(X, y, clf, n=10, test_size=.2, eps=.01):
    all_cross_entropy, all_accuracy = [], []
    for i in tqdm(range(n)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        clf.set_params(random_state=i)            
        clf.fit(X_train, y_train)
        
        all_cross_entropy.append(log_loss(y_test, clf.predict_proba(X_test), eps=eps))
        all_accuracy.append(clf.score(X_test, y_test))
    return all_cross_entropy, all_accuracy


clf = GradientBoostingClassifier(learning_rate=0.015, n_estimators=300, max_depth=6, min_samples_split=30, min_samples_leaf=16)
X = np.load('./train_X_mobilenet_v2.npy')
y = np.load('./train_y_mobilenet_v2.npy')

cen, acc = bootstrap_performance(X, y, clf, n=50)
print(f'mean cross entropy: {np.mean(cen)}')
print(f'mean accuracy: {np.mean(acc)}')