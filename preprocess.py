### Data Preprocessing 시작
print("Preprocessing start")

## 필요 library import


print("Importing libraries...")

import pickle
import pandas as pd
import numpy as np
import h2o
h2o.init()
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.frame import H2OFrame
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

print("All libraries imported.")


## 1.베이스 데이터 전처리


print("Step 1 Proceeding...")

train = pd.read_csv("train.csv")


train['event_datetime']= pd.to_datetime(train['event_datetime']) # event_datetime 변환

train["datetime-day"] = train["event_datetime"].dt.day
train["datetime-hour"] = train["event_datetime"].dt.hour
train["datetime-dayofweek"] = train["event_datetime"].dt.dayofweek

train.loc[train["datetime-dayofweek"] == 0, "weekdays"] = "Monday"
train.loc[train["datetime-dayofweek"] == 1, "weekdays"] = "Tuesday"
train.loc[train["datetime-dayofweek"] == 2, "weekdays"] = "Wednesday"
train.loc[train["datetime-dayofweek"] == 3, "weekdays"] = "Thursday"
train.loc[train["datetime-dayofweek"] == 4, "weekdays"] = "Friday"
train.loc[train["datetime-dayofweek"] == 5, "weekdays"] = "Saturday"
train.loc[train["datetime-dayofweek"] == 6, "weekdays"] = "Sunday"

# 필요없는 변수 제거
train = train.drop(['event_datetime','datetime-day','bid_id','device_country','datetime-dayofweek'], axis=1)
train['datetime-hour'] = train['datetime-hour'].astype(str) # 시간 변수 str 변환

print("Step 1 Done.")


## 2. Audience_Profile 전처리


print("Step 2 Proceeding...")

original_ap = pd.read_csv("audience_profile.csv",sep='!@#')

device_ifa_key = train['device_ifa'].unique() # train data 상에만 존재하는 ap key값
train_ap = original_ap[original_ap['device_ifa'].isin(device_ifa_key)] # key값 기준 train와 ap의 공통값 추출
device_ifa = list(train_ap['device_ifa'])

del(original_ap, device_ifa_key)

print("Step 2 Done.")


## 3. Audience_Profile의 install_pack을 벡터화 하여 cluster


print("Step 3 Proceeding...")

#각 Audience 별로 가지고 있는 app 딕셔너리로
audience_app={}
for i in range(len(train_ap)):
    audience_app[train_ap.iloc[i,0]] = pd.Series(train_ap.iloc[i,4].split(sep=','))

# unique 값 저장
total_app = []
for value in audience_app.values():
    for item in value:
        total_app.append(item)
unique_app = list(set(total_app))

# mapping 할 hash_dict 저장
import hashlib
import math
hash_value = 256
hashSHA = hashlib.sha256()
hash_dict = {}
for val in unique_app:
    hashSHA.update(val.encode('utf-8'))
    hash_dict[val] = str(bin(int(hashSHA.hexdigest(),16))[-(int(math.log(hash_value,2))+1):-1])

#각 Auidence 별 가지고 있는 app의 개수 저장
length = []
for key in audience_app.keys():
    length.append(len(audience_app[key]))
    
#hash dict 이용해서 app을 hash 값으로 변환 
total_hashed_app = pd.Series(total_app).map(hash_dict)

#hashed app unique value 들을 str로 저장
unique_hashed_app_str = list(set(total_hashed_app.astype(str)))

#각 Audience 별로 가지고 있는 hash된 app 딕셔너리로
audience_hashed_app = {}
i=0
j=0
for key in audience_app.keys():
    audience_hashed_app[key] = total_hashed_app[i: i + length[j]].value_counts()
    i = i + length[j]
    j = j + 1
    
#각 Audience 별로 가지고 있는 hash된 app의 개수 저장
dict_hashed_app = {}
for key in audience_app.keys():
    dict_hashed_app[key] = audience_hashed_app[key].to_dict()

#Audience 별 벡터화 된 hashed app을 저장한 DataFrame
app = pd.DataFrame()


for value in unique_hashed_app_str:
    app[value] = [0]*len(train_ap)

for i in range(len(app)):
    app.iloc[i,:] = dict_hashed_app[device_ifa[i]]

app = app.fillna(0).astype(int)

app.insert(loc=0,column='device_ifa',value = device_ifa)

#app kmeans clustering
app_train = H2OFrame(app)
app_cols = app_train.columns
app_cols.remove("device_ifa")

app_kmeans = H2OKMeansEstimator(k = 10, init="Random", standardize = True)
app_kmeans.train(x=app_cols, training_frame = app_train)
app_predicted = app_kmeans.predict(app_train)

pd_app_predicted = app_predicted['predict'].as_data_frame(use_pandas=True, header=True)

app_clust = pd.DataFrame()
app_clust['device_ifa'] = app['device_ifa']
app_clust['app_clusters'] = pd_app_predicted['predict']

app_kmeans_model_path = h2o.save_model(model = app_kmeans, path="./preprocess_model", force = True)

wf = open("app_kmeans_model_path.txt",'wb')
pickle.dump(app_kmeans_model_path, wf)
wf.close()

del(audience_app,total_app,unique_app,total_hashed_app,audience_hashed_app,dict_hashed_app,app,
    app_train,app_cols,app_kmeans,app_predicted,pd_app_predicted)


print("Step 3 Done.")


## 4. Audience_Profile의 cate_code를 벡터화 하여 cluster


print("Step 4 Proceeding...")

#각 Audience 별로 가지고 있는 cate 딕셔너리로
audience_cate = {}
for ifa in device_ifa:
    audience_cate[ifa]={}

for i in range(len(train_ap)):
    for cate in train_ap.iloc[i,5].split(sep=','):
        audience_cate[train_ap.iloc[i,0]][cate[0:5]] = int(cate[-1])

#전체 및 Unique Category
total_cate=[]
for i in range(len(train_ap)):
    for cate in train_ap.iloc[i,5].split(sep=','):
        total_cate.append(cate[0:5])
unique_cate = list(set(total_cate))

wf = open("unique_cate_train.txt",'wb')
pickle.dump(unique_cate, wf)
wf.close()

#Audience 별 벡터화 된 cate_code를 저장한 DataFrame
cate = pd.DataFrame()

for value in unique_cate:
    cate[value] = [0]*len(train_ap)

for i in range(len(cate)):
    cate.iloc[i,:] = audience_cate[device_ifa[i]]

cate = cate.fillna(0).astype(int)
cate.insert(loc=0,column='device_ifa',value = device_ifa)

#cate kmeans clustering
cate_train = H2OFrame(cate)
cate_cols = cate_train.columns
cate_cols.remove("device_ifa")

cate_kmeans = H2OKMeansEstimator(k = 7, init="Random", standardize=True)
cate_kmeans.train(x=cate_cols, training_frame = cate_train)
cate_predicted = cate_kmeans.predict(cate_train)
pd_cate_predicted = cate_predicted['predict'].as_data_frame(use_pandas=True, header=True)

cate_clust = pd.DataFrame()
cate_clust['device_ifa'] = cate['device_ifa']
cate_clust['cate_clusters'] = pd_cate_predicted['predict']

cate_kmeans_model_path = h2o.save_model(model = cate_kmeans, path="./preprocess_model", force = True)

wf = open("cate_kmeans_model_path.txt",'wb')
pickle.dump(cate_kmeans_model_path, wf)
wf.close()

del(audience_cate,total_cate,unique_cate,cate,cate_cols,cate_kmeans,cate_predicted,pd_cate_predicted)


print("Step 4 Done.")


## 5. Audience_Profile의 각 정보를 통해 각 device_ifa 별 predicted mean ctr 계산


print("Step 5 Proceeding...")

#click과 AP 뽑아오기
data_click = train[['click','device_ifa']]
key = pd.DataFrame(train_ap['device_ifa'])

data_ctr = pd.merge(data_click, key, how = 'inner',on = 'device_ifa')

#AP 별 mean ctr 계산
ap_ctr = pd.DataFrame(data_ctr.groupby('device_ifa')['click'].mean())
ap_ctr.reset_index(level=0, inplace = True)

#clust 제외한 columns
ap_ctr_columns = pd.merge(ap_ctr,train_ap, how='inner',on='device_ifa')
ap_ctr_columns = ap_ctr_columns.iloc[:,0:5]

#clust 합치기
ap_ctr_clust = pd.merge(app_clust,cate_clust,how='inner',on='device_ifa')

#columns와 clust 합치기
ap_ctr_final = pd.merge(ap_ctr_columns,ap_ctr_clust, how = 'inner', on = 'device_ifa')

ap_ctr_final.app_clusters = ap_ctr_final.app_clusters.astype(str)
ap_ctr_final.cate_clusters = ap_ctr_final.cate_clusters.astype(str)
ap_ctr_final.age = ap_ctr_final.age.astype(str)

#linear regression으로 ap의 정보를 통해 각 AP 별 predicted mean ctr 계산
ap_ctr_train = H2OFrame(ap_ctr_final)

ap_ctr_cols = ap_ctr_train.columns
ap_ctr_cols.remove('click')
ap_ctr_cols.remove('device_ifa')
response = 'click'

for col in ap_ctr_cols:
    ap_ctr_train[col] = ap_ctr_train[col].asfactor()

lr_model = H2OGeneralizedLinearEstimator(family= "gaussian", lambda_ = 0,nfolds = 5, early_stopping = True)
lr_model.train(x=ap_ctr_cols,y=response,training_frame = ap_ctr_train)

ap_ctr_predicted = lr_model.predict(ap_ctr_train)
pd_ap_ctr_predicted = ap_ctr_predicted['predict'].as_data_frame(use_pandas=True, header=True)
ap_ctr_final['predicted'] = pd_ap_ctr_predicted['predict']

lr_model_path = h2o.save_model(model = lr_model, path="./preprocess_model", force = True)

wf = open("lr_model_path.txt",'wb')
pickle.dump(lr_model_path, wf)
wf.close()

#계산 최종본
predicted_final = ap_ctr_final.loc[:,['device_ifa','predicted']]

wf = open("predicted_final_train.txt","wb")
pickle.dump(predicted_final,wf)
wf.close()

print("Step 5 Done.")


## 6. 최종 Preprocessed 출력


print("Step 6 Proceeding...")

final_train = pd.merge(train,predicted_final,how = "left",on = "device_ifa")
final_train['predicted'][final_train['predicted'].isnull()] = predicted_final.predicted.mean()
final_train['predicted_standard'] = (final_train['predicted'] - predicted_final['predicted'].mean())/predicted_final['predicted'].std()

final_train['predicted_cate'] = ['a'] * final_train.shape[0]

final_train['predicted_cate'][final_train['predicted_standard'] >= 1] = 'VH'
final_train['predicted_cate'][(final_train['predicted_standard'] > 0) & (final_train['predicted_standard'] < 1)] = 'H'
final_train['predicted_cate'][final_train['predicted_standard'] == 0] = 'AVG'
final_train['predicted_cate'][(final_train['predicted_standard'] > -1) & (final_train['predicted_standard'] < 0)] = 'L'
final_train['predicted_cate'][final_train['predicted_standard'] <= -1] = 'VL' 

final_train = final_train.drop(['predicted','predicted_standard'],axis= 1)
final_train['click'] = final_train['click'].astype(int)


del(data_ctr,ap_ctr,ap_ctr_columns,ap_ctr_clust,ap_ctr_final,ap_ctr_train,ap_ctr_cols,ap_ctr_predicted,pd_ap_ctr_predicted,
   lr_model,predicted_final,train_ap,train)

train = final_train.copy()
del(final_train)

print("Step 6 Done.")


## 7. Subset 별로 분할

print("Step 7 Proceeding...")

#OS subset

train_os_1 = train[train['device_os']=='TG14pLUXCY'].reset_index(drop = True)
train_os_rest = train[train['device_os'].isin(['TG14pLUXCY']) == False].reset_index(drop = True)

#Placement subset

train_placement_1 = train[train['placement_type']== 'kIeE1J0KCa'].reset_index(drop = True)
train_placement_2 = train[train['placement_type']=='1pcQ3RJgQt'].reset_index(drop = True)
train_placement_rest = train[train['placement_type'].isin(['kIeE1J0KCa','1pcQ3RJgQt']) == False].reset_index(drop = True)

      
print("Step 7 Done.")

## 8. FFM 형식으로 변환

print("Step 8 Proceeding...")

train_set = [train_os_1,train_os_rest,train_placement_1,train_placement_2,train_placement_rest]
subset_name = ['_os_1','_os_rest','_placement_1','_placement_2','_placement_rest']

# Subset 별 10%를 validation set으로
      
valid_os_1_index = np.random.choice(list(range(train_os_1.shape[0])),int(train_os_1.shape[0]/10), replace=False)
train_os_1_index = np.setdiff1d(np.array(range(train_os_1.shape[0])),valid_os_1_index)
valid_os_rest_index = np.random.choice(list(range(train_os_rest.shape[0])),int(train_os_rest.shape[0]/10), replace=False)
train_os_rest_index = np.setdiff1d(np.array(range(train_os_rest.shape[0])),valid_os_rest_index)

valid_placement_1_index = np.random.choice(list(range(train_placement_1.shape[0])),int(train_placement_1.shape[0]/10), replace=False)
train_placement_1_index = np.setdiff1d(np.array(range(train_placement_1.shape[0])),valid_placement_1_index)
valid_placement_2_index = np.random.choice(list(range(train_placement_2.shape[0])),int(train_placement_2.shape[0]/10), replace=False)
train_placement_2_index = np.setdiff1d(np.array(range(train_placement_2.shape[0])),valid_placement_2_index)
valid_placement_rest_index = np.random.choice(list(range(train_placement_rest.shape[0])),int(train_placement_rest.shape[0]/10), replace=False)
train_placement_rest_index = np.setdiff1d(np.array(range(train_placement_rest.shape[0])),valid_placement_rest_index)

valid_index_set = [valid_os_1_index,valid_os_rest_index,valid_placement_1_index,valid_placement_2_index,valid_placement_rest_index]
train_index_set = [train_os_1_index,train_os_rest_index,train_placement_1_index,train_placement_2_index,train_placement_rest_index]

# FFM 형식으로 변환하는 함수
      
def FFM_Converter(df):
    global cols
    global total
    length = df.shape[0] #df의 행의 길이
    final=[] #return 할 최종 list
    for i in range(length):
        ffm=[]
        df_row = df.iloc[i,:]
        ffm.append(str(df_row['click']))
        field=0
        for col in cols:
            element= '{}:{}:1'.format(field,total[col][df_row[col]])
            ffm.append(element)
            field = field + 1
        final.append(' '.join(ffm))
    return pd.Series(final)

for i in range(len(subset_name)):
    train_set[i]['click'] = train_set[i]['click'].astype(int)
    train_set[i]['datetime-hour'] = train_set[i]['datetime-hour'].astype(str)
    cols = train_set[i].columns #columns list 생성
    cols = cols.drop("click")  #click 제외

    wf = open("cols" + subset_name[i] + ".txt",'wb')
    pickle.dump(cols, wf)
    wf.close()
    
    total = {col:{} for col in cols}
    
    t=0
    for category in total.keys():
        unique_category = list(train_set[i][category].unique())
        unique_category.append('NA')
        for unique in unique_category:
            total[category][unique]=t
            t=t+1

    wf = open("total" + subset_name[i] + ".txt",'wb')
    pickle.dump(total, wf)
    wf.close()
    
    FFM_train = FFM_Converter(train_set[i])
    
    FFM_train_split = FFM_train.iloc[list(train_index_set[i])].reset_index(drop = True)
    FFM_valid_split = FFM_train.iloc[list(valid_index_set[i])].reset_index(drop = True)
    
    FFM_train_txt = open("FFM_train" + subset_name[i] + "_preprocess.txt","w") 
    for row in FFM_train_split:
        FFM_train_txt.write(row + '\n')
    FFM_train_txt.close()
    
    FFM_valid_txt = open("FFM_valid" + subset_name[i] + "_preprocess.txt","w") 
    for row in FFM_valid_split:
        FFM_valid_txt.write(row + '\n')
    FFM_valid_txt.close()


print("Step 8 Done.")


### Data Preprocessing 종료
print("Preprocessing Done.") 
