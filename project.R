#### 심혈관 질환 예측 프로젝트 #### 

# 불러올 패키지들 #
library(tidymodels)
library(dplyr)
library(ggplot2)
library(readr)

# ** basic model ** ####
# cross validation - one decision tree 
# 1. 모델링 준비과정 ####

# 데이터 로드 #
disease_raw <- read_delim("cardiovascular_disease_dataset.csv", delim = ";")

# 데이터 복제
df <- disease_raw

# 데이터 탐색
glimpse(df)
head(df)
tail(df)
dim(df)
summary(df)
structure(df) # gender, cholesterol, gluc, smoke, alco, active 변수 이상치 없음 확인 
View(df)
table(is.na(df)) # 결측치 없음 확인

# 타겟 변수 검토
df %>% 
  count(cardio) %>% 
  mutate(ratio = n/sum(n)*100)
# 50 : 50으로 골고루 있음 확인 
class(df$cardio)

# 타겟변수 전처리
df <- df %>% 
  mutate(cardio = ifelse(cardio ==1, "cardio" , "healthy"),
         cardio = factor(cardio, levels = c("cardio","healthy")))

# 타겟변수 순서 확인
df %>% 
  count(cardio) %>% 
  mutate(ratio = n/sum(n)*100)
levels(df$cardio)

# 변수 검토 및 수정 -> summary(df)
# 아래과정을 안해도 되지만 안하면 순서를 지정해주지 않아서 헷갈릴 가능성이 있기에 factor로 변환해주는게 안전 # 0,1,2 이처럼 나누어져있을 때 범주로 나누어져 있다고 판단하고 자동으로 factor로 변환해서 모델을 만들어주지만, factor이랑 levels를 이용해서 순서를 지정해주지 않으면 헷갈릴 가능성이 있어 levels로 순서 지정해줘야함 -> 이때 levels로 순서 지정해주고 범주형태로 바꿔줌. 
# 0,1이 무엇을 의미하는지 안다면, 결과는 0이 우선이므로 나중에 factor, levels를 안썼을 때 헷갈리지 않도록 조심! -> 어차피 타겟변수의 예측만 잘해주나 관심있으므로 상관없을 듯!
# 예를 들어, 
# df <- df %>% 
#  mutate(smoke = ifelse(smoke ==1, "smoke" , "no"),
#         smoke = factor(smoke, levels = c("smoke","no")))

class(df)
df <- df %>% 
  mutate(age = age/365)
table(df$age)

table(df$gender)
#summary(df$gluc)
# qplot(df$gender)

table(df$cholesterol)
#summary(df$cholesterol)
# qplot(df$cholesterol)

table(df$gluc)
#summary(df$gluc)
# qplot(df$gluc)

table(df$smoke)
#summary(df$smoke)

table(df$alco)

table(df$active)

# 데이터 분할 (8:2)
# cross validation (교차검증) - holdout 이용

set.seed(10)
df_split <- df %>% 
  initial_split(prop = 0.8, strata = cardio)
# -> disease와 healthy 50:50으로 나왔으므로
df_split

df_train <- training(df_split)
df_test <- testing(df_split)

#dim(df_train)
#dim(df_test) 
#glimpse(df_train)
#glimpse(df_test)

# train, test 데이터 검토-> 비율 비슷한지
df_train %>% 
  count(cardio) %>% 
  mutate(ratio = n/sum(n))
df_test %>% 
  count(cardio) %>% 
  mutate(ratio = n/sum(n))

# 2. 모델링 ####

# 레서피
library(dlookr)

# 고유값 많은 변수 찾기
diagnose(df_train) %>% 
  filter(unique_rate > 0.1)
# age는 numeric으로 당연히 고유값 많으므로 변수 포함, id는 제외

stack_recipe <- 
  recipe(cardio ~ ., data = df_train) %>% 
  update_role(id, new_role = "id")

stack_recipe

# 모델 세팅
tree_mod <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# 워크플로우 생성
stack_wf <- workflow() %>% 
  add_recipe(stack_recipe) %>% 
  add_model(tree_mod)

stack_wf

# 모델 적합 - train 데이터
set.seed(10)
fit_tree <- stack_wf %>% 
  fit(data = df_train)

fit_tree

# 변수 중요도 확인
library(vip)
fit_tree %>%
  extract_model() %>%
  vip()

# 예측값(train 데이터 기반) 추가
df_pred_train <- df_train %>% 
  bind_cols(predict(fit_tree, df_train)) %>% 
  rename(pred = .pred_class)
df_pred_train

# 학습데이터 성능평가
conf_train <- conf_mat(data = df_pred_train,
                       truth = cardio,
                       estimate = pred)
conf_train
summary(conf_train)
# accuracy 0.713, precision 0.760, recall = 0.622이므로 정상인을 심혈관 질환으로 분류하고 지켜보는 것보다 심혈관 질환을 놓치는 경우가 더 심각하므로 recall을 주의깊게 봐야한다고 판단

# 예측값(test 데이터 기반) 추가
df_pred_test <- df_test %>% 
  bind_cols(predict(fit_tree,df_test)) %>% 
  rename(pred = .pred_class)

glimpse(df_pred_test)  

# 검증데이터 성능평가 -> 후에 모델이 이게 가장 좋다고 하면인데 이게 가장 좋은 성능을 내는게 아니므로 다음 코드는 실행x (가장 좋은 성능평가를 낼 때 이 코드 실행 후 검증데이터에 대한 성능 확인)
conf_test <- conf_mat(data = df_pred_test,
                      truth = cardio,
                      estimate = pred)
conf_test
summary(conf_test)
# summary(conf_train)
# accuracy는 0.717, precision은 0.763, recall은 0.630, f_meas는 0.690으로 학습데이터 성능평가 할 때보다 검증데이터 성능평가 때 조금 더 정확도가 높게 나옴.


# **K-fold방법 기반 모델 (튜닝 전) **####
set.seed(10)
folds <- vfold_cv(df_train, v = 10)
folds

# folds별 모델 생성 및 예측
set.seed(10)
fit_tree_kfold <- 
  stack_wf %>% 
  fit_resamples(resamples = folds,
                metrics = metric_set(accuracy, precision, recall, f_meas))

fit_tree_kfold

# 개별 fold 살펴보기
fit_tree_kfold %>% 
  filter(id == "Fold01") %>% 
  unnest(.metrics)

# 성능 평가
collect_metrics(fit_tree_kfold)
# accuracy는 0.713, precision은 0.760, f_meas는 0.684, recall은 0.622
# crossvalidation-holdout의 학습데이터 결과와 동일(아래코드)
summary(conf_train) %>%
  slice(1, 13, 11, 12)    # 와 비교

# k-fold 성능평가 결과 개별적으로 살펴보기
tail(collect_metrics(fit_tree_kfold, summarize = F))


# 해야할 과정####
# 튜닝과정#
# 튜닝과정 후 K-fold방법 다시 적용해 교재 뒤에
# 나와있는 하나의 모델로 합쳐서 다시 성능 평가
# 다른 방법론들 이용
# 파생변수 추가-> 상대적인 변수로


#** 튜닝 과정 후 k-fold **####
# 앞서 2. 모델링 목차 부분을 이 코드로 실행

# 하이퍼 파라미터 탐색
# 1) 5- folds ####
set.seed(10)
folds <- vfold_cv(df_train, v = 5)
folds

# 변수 검토
library(dlookr)
diagnose(df_train) %>% 
  filter(unique_rate > 0.1)
# 마찬가지로 age는 고유값이 많은게 당연, id만 제외

# 레시피 생성
stack_recipe <- 
  recipe(data = df_train, cardio ~ .) %>% 
  update_role(id, new_role = "id")

stack_recipe

# 모델 세팅
# 튜닝할 하이퍼 파라미터: cost complexity, tree_depth
tree_spec <- decision_tree(cost_complexity = tune(),
                           tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tree_spec

# 워크플로우 생성
stack_wf_tune <- workflow() %>% 
  add_recipe(stack_recipe) %>% 
  add_model(tree_spec)

stack_wf_tune

# tune grid 생성 -> 5*5 = 25세트
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)
tree_grid

# fold에 tree_grid 적용 모델 생성 -> 5(fold)*5*5 = 125모델
library(tune)
set.seed(10)
fit_tree_tune <- stack_wf_tune %>% 
  tune_grid(resamples = folds,
            grid = tree_grid,
            metrics = metric_set(accuracy, precision, recall, f_meas, roc_auc))

# 하이퍼파라미터 살펴보기 - folds 평균
fit_tree_tune %>% 
  collect_metrics()

# 지표 가장 높은 후보
fit_tree_tune %>% 
  show_best("precision")

fit_tree_tune %>% 
  show_best("accuracy")

fit_tree_tune %>% 
  show_best("roc_auc")

# 최적 선택 - 성능은 높으면서 단순한 모델
best_tree <- fit_tree_tune %>% 
  select_best("roc_auc")
best_tree
# cost_complexity가 0.0000000001, tree_dept = 8

# 최종 모델 생성 - training 데이터
# 현재 워크플로우(stack_wf_tune) 업데이트
final_wf <- stack_wf_tune %>% 
  finalize_workflow(best_tree)
final_wf

# 최종 모델 생성
set.seed(10)
fit_tree_tune_final <- final_wf %>% 
  fit(data = df_train)

fit_tree_tune_final

# k-fold 모델에 대입
# folds별 모델 생성 및 예측
set.seed(10)
fit_tree_tune_kfold <- 
  fit_tree_tune_final %>% 
  fit_resamples(resamples = folds,
                metrics = metric_set(accuracy, precision, recall, f_meas))

fit_tree_tune_kfold

# 개별 fold 살펴보기
fit_tree_tune_kfold %>% 
  filter(id == "Fold01") %>% 
  unnest(.metrics)

# 성능 평가 - 학습데이터
collect_metrics(fit_tree_tune_kfold)
# accuracy는 0.729, precision은 0.753, recall은 0.680, f_meas는 0.715
collect_metrics(fit_tree_kfold) # 튜닝전의 k-fold모델과 비교 ->0.713, 0.760,0.622,0.684
summary(conf_train) %>%
  slice(1, 13, 11, 12)    # 베이직 모델과 비교 -> 0.713, 0.760, 0.622, 0.684
# 결과적으로, 튜닝했을 때의 결과, 더 성능이 높다고 할 수 있음.

# k-fold 성능평가 결과 개별적으로 살펴보기
tail(collect_metrics(fit_tree_kfold, summarize = F))


# ** random forest 방법 이용 ->를 basic모델로 ** ####
# 레서피 생성 - imputation : step_knnimpute()
recipe_rf <- 
  recipe(data = df, cardio ~ .) %>% 
  update_role(id,new_role = "id") %>% 
  step_knnimpute(all_predictors())

# 모델 세팅
tree_mod_rf <- rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

# 워크플로우 생성
stack_wf_rf <- workflow() %>% 
  add_recipe(recipe_rf) %>% 
  add_model(tree_mod_rf)

# 모델 생성
set.seed(10)
fit_tree_rf <- stack_wf_rf %>% 
  fit(data = df_train)

# 예측값(train 데이터 기반) 추가
df_pred_train_rf <- df_train %>% 
  bind_cols(predict(fit_tree_rf, df_train)) %>% 
  rename(pred = .pred_class)
df_pred_train_rf

# 학습데이터 성능평가
conf_train_rf <- conf_mat(data = df_pred_train_rf,
                       truth = cardio,
                       estimate = pred)
conf_train_rf
summary(conf_train_rf) %>% 
  slice(1,11,12,13)
# accuracy는 0.860, precision은 0.883, recall은 0.831, f_meas는 0.856

# 앞선 방법들과의 학습데이터에 대한 성능평가 비교 # 
collect_metrics(fit_tree_tune_kfold) # 튜닝한 k-fold
collect_metrics(fit_tree_kfold) # 튜닝전의 k-fold
summary(conf_train) %>%
  slice(1, 13, 11, 12)    # basic모델- decision_tree

# 결과적으로, randomforest방법이 가장 성능평가 좋음

# 여기까지의 방법들 중 최종적으로 randomforest 선택
# randomforest방법 검증데이터에 대한 성능평가

# 예측값(test 데이터 기반) 추가
df_pred_test_rf <- df_test %>% 
  bind_cols(predict(fit_tree_rf,df_test)) %>% 
  rename(pred = .pred_class)

glimpse(df_pred_test_rf)  

# 검증데이터 성능평가 -> 후에 모델 확정되면
conf_test_rf <- conf_mat(data = df_pred_test_rf,
                      truth = cardio,
                      estimate = pred)
conf_test_rf
summary(conf_test_rf) %>% 
  slice(1,11,12,13)
# accuracy : 0.731, precision : 0.745, recall : 0.702, f_meas : 0.723


# 후에 더 할일 # 
# k-fold 튜닝할 때 하이퍼파라미터, k값 바꿔서 해보기
# 파생변수 추가 등

# 부연설명# 
# 방법을 basic(decision_tree한번), k-fold, tuning k-fold만 했을 때
# randomforest 하기 전까지는 튜닝한 k-fold방법이 가장 성능 좋음
# 튜닝한 k-fold방법을 확정된 모델로 할 때 검증데이터 성능평가 결과 
glimpse(df_test)
df_pred_test_tune <- df_test %>% 
  bind_cols(predict(fit_tree_tune_final, df_test)) %>% 
  rename(pred = .pred_class)

glimpse(df_pred_test_tune)
conf_test_tune <- conf_mat(data = df_pred_test_tune,
                           truth = cardio,
                           estimate = pred)
conf_test_tune
summary(conf_test_tune) %>% 
  slice(1,11,12,13)
# accuracy : 0.734, precision: 0.741, recall : 0.719, f_meas : 0.730