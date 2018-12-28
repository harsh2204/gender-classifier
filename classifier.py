from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier #K Nearest Neighbor
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.naive_bayes import GaussianNB # Naive Bayes


tree_clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
kneighbo_clf = KNeighborsClassifier(n_neighbors=3)
# 2
logistic_reg = LogisticRegression()
# 3
gnb = GaussianNB()

    # [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
tree_clf = tree_clf.fit(X, Y)

kneighbo_clf = kneighbo_clf.fit(X, Y)

logistic_reg = logistic_reg.fit(X, Y)

gnb = gnb.fit(X,Y)


prediction_tree = tree_clf.predict([[190, 70, 43]])
prediction_neigh = kneighbo_clf.predict([[190, 70, 43]])
prediction_logist = logistic_reg.predict([[190, 70, 43]])
prediction_gnb = gnb.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(f'Predition tree : {prediction_tree}')           #male
print(f'KNearest Neighbor: {prediction_neigh}')        #male
print(f'Logistic Regression: {prediction_logist}')     #female   (wrong!)
print(f'Naive Bayes: {prediction_gnb}')                #male