import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data=load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Model acuracy:{accuracy*100:.2f}%")
new_sample=np.array([X_test[0]])
prediction=clf.predict(new_sample)

predicted_class="BENIGN" if prediction==1 else "MALIGNANT"
print(f"Predicted class:{predicted_class}")
plt.figure(figsize=(10,8))
tree.plot_tree(clf,filled=True,feature_names=data.feature_names,class_names=data.target_names)
plt.title("DecisionTree")
plt.show()
