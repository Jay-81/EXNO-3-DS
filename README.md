## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
~~~


<img width="352" height="434" alt="image" src="https://github.com/user-attachments/assets/ba99f1ad-6ca1-4b48-8f20-d5d9e03e21e4" />


~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~

<img width="146" height="221" alt="image" src="https://github.com/user-attachments/assets/6d8d045a-bab9-42fb-9fb9-088b1552d459" />


~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~

<img width="386" height="434" alt="image" src="https://github.com/user-attachments/assets/28aefd65-22b3-4dae-b3ad-724291562f65" />

~~~

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

~~~


<img width="392" height="430" alt="image" src="https://github.com/user-attachments/assets/adaed87d-63a5-408e-9cae-70db16037d49" />

~~~

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

one = OneHotEncoder(sparse_output=False)   
df2 = df.copy()
enc = pd.DataFrame(one.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
~~~

<img width="502" height="432" alt="image" src="https://github.com/user-attachments/assets/8463c76e-20c0-4b2f-b7e6-1f3346f96494" />

~~~

pd.get_dummies(df2,columns=["nom_0"])

~~~

<img width="776" height="430" alt="image" src="https://github.com/user-attachments/assets/231f0a88-89ed-4dfd-95bc-b2a173f180f9" />

~~~
pip install --upgrade category_encoders
~~~

<img width="1557" height="425" alt="image" src="https://github.com/user-attachments/assets/931ee541-8343-4c6c-9240-3d32d1b4d8cc" />

~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
~~~

<img width="825" height="446" alt="image" src="https://github.com/user-attachments/assets/8dc1e098-7eda-487b-8ff8-7f74f6bcc747" />

~~~

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~

<img width="667" height="423" alt="image" src="https://github.com/user-attachments/assets/03b78b88-b5f6-4608-8d67-c8708fd9fb35" />

~~~

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

~~~

<img width="939" height="504" alt="image" src="https://github.com/user-attachments/assets/41b96318-9a3f-4a83-a6bc-50791d711002" />

~~~
df.skew()
~~~

<img width="372" height="245" alt="image" src="https://github.com/user-attachments/assets/2bcec77d-cc4a-48c1-9c50-63507d63df5f" />

~~~
np.log(df["Highly Positive Skew"])
~~~


<img width="273" height="499" alt="image" src="https://github.com/user-attachments/assets/d8b24289-e0fc-4e5e-98bf-bda6003a26b9" />


~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~

<img width="306" height="550" alt="image" src="https://github.com/user-attachments/assets/1727fdc1-b80b-42da-a1af-ed30c0f4ebbe" />


~~~
np.sqrt(df["Highly Positive Skew"])
~~~

<img width="352" height="597" alt="image" src="https://github.com/user-attachments/assets/2220e765-a091-4cc4-8285-91515c18d149" />

~~~
np.square(df["Highly Positive Skew"])
~~~

<img width="332" height="553" alt="image" src="https://github.com/user-attachments/assets/ef83d4b8-afa1-4ce2-9288-cf47786d2606" />

~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~

<img width="1241" height="517" alt="image" src="https://github.com/user-attachments/assets/1ae07df9-a56a-4eba-b419-508d8b92b27c" />

~~~
df.skew()
~~~

<img width="461" height="325" alt="image" src="https://github.com/user-attachments/assets/a953e19b-bba6-47ec-8d30-6b42e18b660e" />

~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~

<img width="468" height="326" alt="image" src="https://github.com/user-attachments/assets/ccea007b-178a-4486-8e35-b51944bbe528" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~

<img width="1718" height="546" alt="image" src="https://github.com/user-attachments/assets/3ed97f7a-1092-4e36-bf10-6d9e15f28319" />

~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~

<img width="732" height="536" alt="image" src="https://github.com/user-attachments/assets/71e3348a-fe48-4779-b739-e08a9612b68f" />

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~

<img width="716" height="529" alt="image" src="https://github.com/user-attachments/assets/e55ae036-8094-42a3-a7d0-498ead58e533" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~

<img width="704" height="541" alt="image" src="https://github.com/user-attachments/assets/d91ccf8d-0108-48c9-94fa-fe25ad406d0e" />

~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~

<img width="709" height="536" alt="image" src="https://github.com/user-attachments/assets/e767300e-8057-4a85-9d54-b453ec214cb3" />

~~~
dt=pd.read_csv("data.csv")
dt
~~~

<img width="579" height="441" alt="image" src="https://github.com/user-attachments/assets/7ede8bb6-16b4-4109-888e-9e4b5c1f4bdd" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Ord_1"]=qt.fit_transform(dt[["Target"]])
sm.qqplot(dt['Target'],line='45') 
plt.show()
~~~


<img width="1306" height="577" alt="image" src="https://github.com/user-attachments/assets/a6f5e6f4-ff3d-4e59-9247-4808dcba310c" />


~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~



<img width="787" height="564" alt="image" src="https://github.com/user-attachments/assets/dd2fae8b-1945-480a-856b-28c7d686d12d" />









      
# RESULT:

    Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
