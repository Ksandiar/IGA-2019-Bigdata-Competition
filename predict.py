#!/usr/bin/env python
# coding: utf-8

### Test Data Preprocessing 시작
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

test = pd.read_csv("test.csv")

test['event_datetime']= pd.to_datetime(test['event_datetime']) # event_datetime 변환

test["datetime-day"] = test["event_datetime"].dt.day
test["datetime-hour"] = test["event_datetime"].dt.hour
test["datetime-dayofweek"] = test["event_datetime"].dt.dayofweek

test.loc[test["datetime-dayofweek"] == 0, "weekdays"] = "Monday"
test.loc[test["datetime-dayofweek"] == 1, "weekdays"] = "Tuesday"
test.loc[test["datetime-dayofweek"] == 2, "weekdays"] = "Wednesday"
test.loc[test["datetime-dayofweek"] == 3, "weekdays"] = "Thursday"
test.loc[test["datetime-dayofweek"] == 4, "weekdays"] = "Friday"
test.loc[test["datetime-dayofweek"] == 5, "weekdays"] = "Saturday"
test.loc[test["datetime-dayofweek"] == 6, "weekdays"] = "Sunday"

# 필요없는 변수 제거
test = test.drop(['event_datetime','datetime-day','bid_id','device_country','datetime-dayofweek'], axis=1)
test['datetime-hour'] = test['datetime-hour'].astype(str) # 시간 변수 str 변환

print("Step 1 Done.")


## 2. Audience_Profile 전처리


print("Step 2 Proceeding...")

original_ap = pd.read_csv("audience_profile.csv",sep='!@#')

device_ifa_key = test['device_ifa'].unique() # test data 상에 존재하는 ap key값
test_ap_whole = original_ap[original_ap['device_ifa'].isin(device_ifa_key)] # key값 기준 test와 ap의 공통값 추출

del(original_ap, device_ifa_key)

print("Step 2 Done.")



rf = open("predicted_final_train.txt","rb")
predicted_final_train = pickle.load(rf)
rf.close()



test_only_device = (set(test_ap_whole['device_ifa'])).difference(set(predicted_final_train['device_ifa']))
test_only_device = pd.Series(list(test_only_device))




test_ap = test_ap_whole[test_ap_whole['device_ifa'].isin(test_only_device)]
test_ap = test_ap.reset_index(drop=True)




device_ifa = list(test_ap.device_ifa)




## 3. Audience_Profile의 install_pack을 벡터화 하여 cluster


print("Step 3 Proceeding...")

#각 Audience 별로 가지고 있는 app 딕셔너리로
audience_app={}
for i in range(len(test_ap)):
    audience_app[test_ap.iloc[i,0]] = pd.Series(test_ap.iloc[i,4].split(sep=','))




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
    app[value] = [0]*len(test_ap)

for i in range(len(app)):
    app.iloc[i,:] = dict_hashed_app[device_ifa[i]]

app = app.fillna(0).astype(int)

app.insert(loc=0,column='device_ifa',value = device_ifa)




#app kmeans clustering
app_train = H2OFrame(app)
app_cols = app_train.columns
app_cols.remove("device_ifa")



rf = open("app_kmeans_model_path.txt","rb")
app_kmeans_model_path = pickle.load(rf)
rf.close()



app_kmeans = h2o.load_model(app_kmeans_model_path)



app_predicted = app_kmeans.predict(app_train)



pd_app_predicted = app_predicted['predict'].as_data_frame(use_pandas=True, header=True)

app_clust = pd.DataFrame()
app_clust['device_ifa'] = app['device_ifa']
app_clust['app_clusters'] = pd_app_predicted['predict']



del(audience_app,total_app,unique_app,total_hashed_app,audience_hashed_app,dict_hashed_app,app,
    app_train,app_cols,app_kmeans,app_predicted,pd_app_predicted)


print("Step 3 Done.")



## 4. Audience_Profile의 cate_code를 벡터화 하여 cluster


print("Step 4 Proceeding...")

#각 Audience 별로 가지고 있는 cate 딕셔너리로
audience_cate = {}
for ifa in device_ifa:
    audience_cate[ifa]={}

for i in range(len(test_ap)):
    for cate in test_ap.iloc[i,5].split(sep=','):
        audience_cate[test_ap.iloc[i,0]][cate[0:5]] = int(cate[-1])

#전체 및 Unique Category
total_cate=[]
for i in range(len(test_ap)):
    for cate in test_ap.iloc[i,5].split(sep=','):
        total_cate.append(cate[0:5])
unique_cate = list(set(total_cate))



#Audience 별 벡터화 된 cate_code를 저장한 DataFrame
cate = pd.DataFrame()

for value in unique_cate:
    cate[value] = [0]*len(test_ap)

for i in range(len(cate)):
    cate.iloc[i,:] = audience_cate[device_ifa[i]]

cate = cate.fillna(0).astype(int)

rf = open("unique_cate_train.txt","rb")
unique_cate_train = pickle.load(rf)
rf.close()

cate = cate[list(unique_cate_train)]

cate.insert(loc=0,column='device_ifa',value = device_ifa)



#cate kmeans clustering
cate_train = H2OFrame(cate)
cate_cols = cate_train.columns
cate_cols.remove("device_ifa")



rf = open("cate_kmeans_model_path.txt","rb")
cate_kmeans_model_path = pickle.load(rf)
rf.close()
cate_kmeans = h2o.load_model(cate_kmeans_model_path)



cate_predicted = cate_kmeans.predict(cate_train)



pd_cate_predicted = cate_predicted['predict'].as_data_frame(use_pandas=True, header=True)

cate_clust = pd.DataFrame()
cate_clust['device_ifa'] = cate['device_ifa']
cate_clust['cate_clusters'] = pd_cate_predicted['predict']

del(audience_cate,total_cate,unique_cate,cate,cate_cols,cate_kmeans,cate_predicted,pd_cate_predicted)


print("Step 4 Done.")



## 5. Audience_Profile의 각 정보를 통해 각 device_ifa 별 predicted mean ctr 계산

test_ctr_final = test_ap.iloc[:,0:4]
test_ctr_final['app_clusters'] = app_clust['app_clusters']
test_ctr_final['cate_clusters'] = cate_clust['cate_clusters']



test_ctr_final.app_clusters = test_ctr_final.app_clusters.astype(str)
test_ctr_final.cate_clusters = test_ctr_final.cate_clusters.astype(str)
test_ctr_final.age = test_ctr_final.age.astype(str)



rf = open("lr_model_path.txt","rb")
lr_model_path = pickle.load(rf)
rf.close()



lr_model = h2o.load_model(lr_model_path)


test_ctr_train = H2OFrame(test_ctr_final)

test_ctr_cols = test_ctr_train.columns
test_ctr_cols.remove('device_ifa')

for col in test_ctr_cols:
    test_ctr_train[col] = test_ctr_train[col].asfactor()

test_ctr_predicted = lr_model.predict(test_ctr_train)
pd_test_ctr_predicted = test_ctr_predicted['predict'].as_data_frame(use_pandas=True, header=True)
test_ctr_final['predicted'] = pd_test_ctr_predicted['predict']



#계산 최종본
predicted_final = test_ctr_final.loc[:,['device_ifa','predicted']]

print("Step 5 Done.")



intersected_device = set(predicted_final_train.device_ifa).intersection(test_ap_whole.device_ifa)
intersected_ctr = predicted_final_train[predicted_final_train['device_ifa'].isin(intersected_device)]
predicted_final_whole = pd.concat([predicted_final,intersected_ctr],ignore_index = True)



## 6. 최종 Preprocessed csv 출력


print("Step 6 Proceeding...")

final_test = pd.merge(test,predicted_final_whole,how = "left",on = "device_ifa")
final_test['predicted'][final_test['predicted'].isnull()] = predicted_final_train.predicted.mean()
final_test['predicted_standard'] = (final_test['predicted'] - predicted_final_train['predicted'].mean())/predicted_final_train['predicted'].std()

final_test['predicted_cate'] = ['a'] * final_test.shape[0]

final_test['predicted_cate'][final_test['predicted_standard'] >= 1] = 'VH'
final_test['predicted_cate'][(final_test['predicted_standard'] > 0) & (final_test['predicted_standard'] < 1)] = 'H'
final_test['predicted_cate'][final_test['predicted_standard'] == 0] = 'AVG'
final_test['predicted_cate'][(final_test['predicted_standard'] > -1) & (final_test['predicted_standard'] < 0)] = 'L'
final_test['predicted_cate'][final_test['predicted_standard'] <= -1] = 'VL' 

final_test = final_test.drop(['predicted','predicted_standard'],axis= 1)

del(test_ctr_final,test_ctr_cols,test_ctr_train,test_ctr_predicted,pd_test_ctr_predicted,
   lr_model,predicted_final,test_ap,test,test_ap_whole)

test = final_test.copy()
del(final_test)

print("Step 6 Done.")


## 7. Subset 별로 분할

print("Step 7 Proceeding...")

#OS subset

test_os_1 = test[test['device_os']=='TG14pLUXCY'].reset_index(drop = True)
test_os_rest = test[test['device_os'].isin(['TG14pLUXCY']) == False].reset_index(drop = True)


#Placement subset

test_placement_1 = test[test['placement_type']== 'kIeE1J0KCa'].reset_index(drop = True)
test_placement_2 = test[test['placement_type']=='1pcQ3RJgQt'].reset_index(drop = True)
test_placement_rest = test[test['placement_type'].isin(['kIeE1J0KCa','1pcQ3RJgQt']) == False].reset_index(drop = True)
      
print("Step 7 Done.")


## 8. FFM 형식으로 변환

print("Step 8 Proceeding...")

test_set = [test_os_1,test_os_rest,test_placement_1,test_placement_2,test_placement_rest]
subset_name = ['_os_1','_os_rest','_placement_1','_placement_2','_placement_rest']


# FFM 형식으로 변환하는 함수
      
def FFM_Converter_test(df):
    global cols
    global total
    length = df.shape[0] #df의 행의 길이
    final=[] #return 할 최종 list
    for i in range(length):
        ffm=[]
        df_row = df.iloc[i,:]
        field=0
        for col in cols:
            element= '{}:{}:1'.format(field,total[col][df_row[col]])
            ffm.append(element)
            field = field + 1
        final.append(' '.join(ffm))
    return pd.Series(final)

for i in range(len(subset_name)):
    test_set[i]['datetime-hour'] = test_set[i]['datetime-hour'].astype(str)

    rf = open("total"+ subset_name[i] +".txt","rb")
    total = pickle.load(rf)
    rf.close()

    rf = open("cols"+ subset_name[i] +".txt","rb")
    cols = pickle.load(rf)
    rf.close()
    
    for col in cols:
        test_set[i][col][test_set[i][col].isin(list(total[col].keys())) == False] = 'NA'
    
    FFM_test = FFM_Converter_test(test_set[i])
    
    FFM_test_txt = open("FFM_test" + subset_name[i] + "_preprocess.txt","w") 
    for row in FFM_test:
        FFM_test_txt.write(row + '\n')
    FFM_test_txt.close()
    

print("Step 8 Done.")


### Data Preprocessing 종료
print("Preprocessing Done.")


### Prediction
print("Prediction Start")

## FFM prediction
import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTest("FFM_test_os_1_preprocess.txt")
ffm_model.predict("os_1_model.out","os_1_output.txt")
del(ffm_model)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTest("FFM_test_os_rest_preprocess.txt")
ffm_model.predict("os_rest_model.out","os_rest_output.txt")
del(ffm_model)


ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTest("FFM_test_placement_1_preprocess.txt")
ffm_model.predict("placement_1_model.out","placement_1_output.txt")
del(ffm_model)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTest("FFM_test_placement_2_preprocess.txt")
ffm_model.predict("placement_2_model.out","placement_2_output.txt")
del(ffm_model)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTest("FFM_test_placement_rest_preprocess.txt")
ffm_model.predict("placement_rest_model.out","placement_rest_output.txt")
del(ffm_model)
      
print("Prediction done. Move to making submission file")
      
test = pd.read_csv("test.csv")

test_os_1 = test[test['device_os']=='TG14pLUXCY'].reset_index(drop=True)
test_os_rest = test[test['device_os'].isin(['TG14pLUXCY']) == False].reset_index(drop=True)

test_placement_1 = test[test['placement_type']== 'kIeE1J0KCa'].reset_index(drop=True)
test_placement_2 = test[test['placement_type']=='1pcQ3RJgQt'].reset_index(drop=True)
test_placement_rest = test[test['placement_type'].isin(['kIeE1J0KCa','1pcQ3RJgQt']) == False].reset_index(drop=True)

#Subset별로 통합
os_1_result = pd.read_csv("os_1_output.txt",names=['prob'])
os_rest_result = pd.read_csv("os_rest_output.txt",names=['prob'])

os_1_sub = pd.DataFrame()
os_1_sub['bid_id'] = test_os_1['bid_id']
os_1_sub['os_prob'] = os_1_result['prob']

os_rest_sub = pd.DataFrame()
os_rest_sub['bid_id'] = test_os_rest['bid_id']
os_rest_sub['os_prob'] = os_rest_result['prob']

os_sub = pd.concat([os_1_sub,os_rest_sub],axis=0,ignore_index = True)


placement_1_result = pd.read_csv("placement_1_output.txt",names=['prob'])
placement_2_result = pd.read_csv("placement_2_output.txt",names=['prob'])
placement_rest_result = pd.read_csv("placement_rest_output.txt",names=['prob'])

placement_1_sub = pd.DataFrame()
placement_1_sub['bid_id'] = test_placement_1['bid_id']
placement_1_sub['placement_prob'] = placement_1_result['prob']

placement_2_sub = pd.DataFrame()
placement_2_sub['bid_id'] = test_placement_2['bid_id']
placement_2_sub['placement_prob'] = placement_2_result['prob']

placement_rest_sub = pd.DataFrame()
placement_rest_sub['bid_id'] = test_placement_rest['bid_id']
placement_rest_sub['placement_prob'] = placement_rest_result['prob']

placement_sub = pd.concat([placement_1_sub,placement_2_sub,placement_rest_sub],axis=0,ignore_index = True)

#앙상블(로지스틱 평균)
ensenble = pd.merge(os_sub,placement_sub,how = 'inner',on='bid_id')

def logit(x):
    return 1/(1+np.exp(-x))

def reverse_logit(x):
    return np.log((x /(1 - x)))

ensenble['mean'] = list(logit((reverse_logit(np.array(ensenble['os_prob'])) + reverse_logit(np.array(ensenble['placement_prob'])))/2))

ensenble_sub = ensenble[['bid_id','mean']]

ensenble_sub.to_csv("predict.csv",index = None, header = None)

print("Predict Done.")














