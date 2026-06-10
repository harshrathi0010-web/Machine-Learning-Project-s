import pandas as pd

# Loading datset
df=pd.read_csv("dataset/Iris.csv")

# Drop unwanted columns
df=df.drop(columns=['Id'])

# Removing missing value 
df=df.dropna()

# DATA encoding change text value(species) into numeric value
df['Species']=df['Species'].map(lambda val:list(df.Species.unique()).index(val))

# Data split in features and labels
X=df.drop(columns=['Species'])
y=df['Species']

# DATA Split in Train - Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

# MODEL selection
from sklearn.ensemble  import RandomForestClassifier
model=RandomForestClassifier()

# MODEL training
model.fit(X_train,y_train)

# MODEL prediction
y_pred=model.predict(X_test)

# MODEL evaluation
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print('accuracy_score:',acc*100)

# Save MODEL
import joblib
joblib.dump( model , "model/model.pkl" )




# Save the MODEL evalutaion
file=open("evaluation.txt",'a')
data=str(acc*100)+'\n' +str(cm)+ '\n' +str(cr)
file.write(data)
file.close()

print("model training and evalution")










