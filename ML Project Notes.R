# Machine Learning Project
# Boston Housing Market
# House Prices: Advanced Regression Techniques

# -------------------------------------------------------------------------------------
#      NOTES 
# -------------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)

# setwd("~/Desktop/NYC DSA/Lecture Slides/Projects/Machine Learning")

df_train = read.csv('train.csv')
df_test = read.csv('test.csv')
df_sample = read.csv('sample_submission.csv')


#  ----------------------------------------------------------------------------------
df_train %>% group_by(Street) %>% summarise(n())
df_train %>% select(c(SalePrice, Street)) %>% ggplot() +geom_boxplot(aes(x=Street, y = SalePrice))
df_train %>% group_by(Street, MSSubClass) %>% summarise(n())
# Only 6 houses have gravel road access to the property out of 1460 houses, therefore, the Street column can be dropped
df_train = df_train %>% select(-c(Street))
# Removed street variable


