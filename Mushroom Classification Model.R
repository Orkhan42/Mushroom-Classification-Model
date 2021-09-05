library(h2o)
library(rstudioapi)
library(highcharter)
library(data.table)
library(tidyverse)
library(inspectdf)
library(caret)

path <- dirname(getSourceEditorContext()$path)
setwd(path)

dataset <- fread("mushrooms.csv")
dataset %>% colnames()

names(dataset) <- names(dataset) %>% str_replace_all("-","_") %>% 
                  str_replace_all("\\%","_")

dataset$class %>% table() %>% prop.table()

sum(is.na(dataset))

colnm <- dataset %>% colnames()

for (i in colnm){
  dataset[[i]] <- dataset[[i]] %>% str_replace_all("'","") %>% as.factor()
}


dataset$class <- dataset$class %>% factor(levels = c('p','e'),
                                          labels = c(1,0))


dataset$class %>% table %>% prop.table() %>% round(2)

#-----------------MODELING------------------------

h2o.init()

h2o_data <- dataset %>% as.h2o()

h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- dataset %>% select(-class) %>% names()

model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  balance_classes = T,
  nfolds = 10, seed = 123,
  max_runtime_secs = 480)


model@leaderboard %>% as.data.frame()
model@leader 

pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()

threshold <- model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1')

confmat <- model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t()


paste("Accuracy = ",
      round(sum(diag(confmat))/sum(confmat)*100,1),"%")


model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))




auc <- model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc()

gini <- model@leader %>% 
  h2o.performance(test) %>% 
  h2o.giniCoef()

accuracy <- model@leader %>% 
  h2o.performance(test) %>% 
  h2o.accuracy()

