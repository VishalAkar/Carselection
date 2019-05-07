import pandas as pd

accuracy_data = []
i = 1
count = 0
#Importing data set
car_Data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data ')

#converting string data into numeric data equivalent
car_Data.replace(to_replace=["5more","more","low","med","high","vhigh","small","big"], value = ['5','5','1','2','3','4','1','3'], inplace = True)

#Creating feature dataset
parameter = car_Data.iloc[:,0:6].values
target = car_Data.iloc[:,6].values

#Splitting training and testing data set
from sklearn.cross_validation import train_test_split 
parameter_train,parameter_test,target_train,target_test = train_test_split(parameter,target, test_size = 0.3, random_state = 0 )

while(True):
    #Feeding data into decision tree 
    from sklearn.tree import DecisionTreeClassifier
    Train = DecisionTreeClassifier( random_state = 0, max_depth = i)
    Train.fit(parameter_train,target_train)
    target_predict = Train.predict(parameter_test)
    
    #Accuracy check
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(target_test,target_predict)
    print("Total Accuracy at Max_depth",i, ": ", accuracy)
    print('misclassified samples %d'%(target_predict != target_test).sum())
    accuracy_data.append(accuracy)
    if i==1:
        best = accuracy_data[0]
        i += 1
    else:
        if best<accuracy:
            best = accuracy
            i +=1
        elif(count<=3):
            i += 1
            count +=1 
        elif(count>3):
            break

print("\n\n")
print("Accuracy at Max_ depth = 3 is :", accuracy_data[2])
print("Best Accuracy is at Max_Depth ",accuracy_data.index(best)+1," and value is ",best*100,"%")