   
def classify(features_train, labels_train):   
    
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn import tree
    
    ### create classifier
    #clf = GaussianNB()
    #clf = SVC(kernel = 'rbf', C = 1000)
    clf = tree.DecisionTreeClassifier(min_samples_split = 50)
    
    ### fit the classifier on the training features and labels    
    clf.fit(features_train,labels_train)
    
    ### return the fit classifier    
    return clf


def accuracy(features_train, labels_train, features_test, labels_test):
    
    ### calculate and return the accuracy on the test data
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn import tree
    
    ### create classifier
    #clf = GaussianNB()
    #clf = SVC(kernel = 'rbf', C = 1000)
    clf = tree.DecisionTreeClassifier(min_samples_split = 50)
    
    ### fit the classifier on the training features and labels    
    clf.fit(features_train,labels_train)
    
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    return accuracy