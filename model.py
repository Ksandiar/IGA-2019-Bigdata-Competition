### 모델 생성 시작

## Subset 별 FFM 모델 생성

import xlearn as xl

print("Creating Models...")

# OS

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTrain("FFM_train_os_1_preprocess.txt")
ffm_model.setValidate("FFM_valid_os_1_preprocess.txt")
param = {'task':'binary','lr':0.1,'k':13,'opt':'adagrad','lambda':0}
ffm_model.fit(param,"os_1_model.out")
del(ffm_model,param)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTrain("FFM_train_os_rest_preprocess.txt")
ffm_model.setValidate("FFM_valid_os_rest_preprocess.txt")
param = {'task':'binary','lr':0.2,'k':12,'opt':'adagrad','lambda':0}
ffm_model.fit(param,"os_rest_model.out")
del(ffm_model,param)
print("OS models created.")
# Placement

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTrain("FFM_train_placement_1_preprocess.txt")
ffm_model.setValidate("FFM_valid_placement_1_preprocess.txt")
param = {'task':'binary','lr':0.1,'k':11,'opt':'adagrad','lambda':0}
ffm_model.fit(param,"placement_1_model.out")
del(ffm_model,param)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTrain("FFM_train_placement_2_preprocess.txt")
ffm_model.setValidate("FFM_valid_placement_2_preprocess.txt")
param = {'task':'binary','lr':0.1,'k':10,'opt':'adagrad','lambda':0}
ffm_model.fit(param,"placement_2_model.out")
del(ffm_model,param)

ffm_model = xl.create_ffm()
ffm_model.setSigmoid()
ffm_model.setTrain("FFM_train_placement_rest_preprocess.txt")
ffm_model.setValidate("FFM_valid_placement_rest_preprocess.txt")
param = {'task':'binary','lr':0.2,'k':12,'opt':'adagrad','lambda':0}
ffm_model.fit(param,"placement_rest_model.out")
del(ffm_model,param)

print("Placement models created.")

print("All models created.")
