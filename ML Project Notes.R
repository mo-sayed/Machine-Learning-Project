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
library(forcats)
library(data.table)
library(VIM)

setwd("~/Desktop/NYC DSA/Lecture Slides/Projects/Machine Learning")

df_train = read.csv('train.csv')
df_test = read.csv('test.csv')
df_sample = read.csv('sample_submission.csv')


#  ----------------------------------------------------------------------------------
# Check for 'MSSubClass' as a predictor
df_train %>% group_by(MSSubClass) %>% summarise(n())
df_train %>% select(c(SalePrice, MSSubClass)) %>% group_by(MSSubClass) %>% summarise(avg_SP=mean(SalePrice)) %>% 
  ggplot() +geom_point(aes(x=MSSubClass, y = avg_SP))
model_MSSubClass = lm(SalePrice ~ MSSubClass, data = df_train)
summary(model_MSSubClass)
# Linear Reg model for MSSubClass has a p-value of 0.001266 but a very low R-Squared
# so can keep it for later EDA

#  ----------------------------------------------------------------------------------
# Check for 'MSZoning' as a predictor
df_train %>% group_by(MSZoning) %>% summarise(n())
df_train %>% select(c(SalePrice, MSZoning)) %>% ggplot() +geom_boxplot(aes(x=MSZoning, y = SalePrice))
df_train %>% select(c(SalePrice, MSZoning)) %>% group_by(MSZoning) %>% summarise(avg_SP=mean(SalePrice)) %>% 
  ggplot() +geom_boxplot(aes(x=MSZoning, y = avg_SP))
model_MSSubClass = lm(SalePrice ~ MSZoning, data = df_train)
summary(model_MSSubClass)
# combine two factors 'RH' and 'RM' together into a single 'RH' to show high residential density
# as the distribution of the SalePrice as a factor of MSZoning is the same for RH & RM
df_train = df_train %>% mutate(MSZoning = ifelse(MSZoning == "C (all)", "C (all)", 
            ifelse(MSZoning == "FV", "FV",ifelse(MSZoning == "RL", "RL", "RH"))))


#  ----------------------------------------------------------------------------------
# Check for 'LotFrontage' as a predictor
df_train %>% group_by(LotFrontage) %>% summarise(n()) %>% arrange(desc(LotFrontage))
df_train %>% select(c(SalePrice, LotFrontage)) %>% ggplot() +geom_point(aes(x=LotFrontage, y = SalePrice))
df_train %>% filter(LotFrontage >250) %>% select(c(SalePrice, LotArea, LotFrontage, TotalBsmtSF, Id))
# df_train = df_train %>% mutate(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage))%>% filter(LotFrontage <250)
df_train %>% filter(is.na(LotFrontage))%>% select(c(SalePrice, LotArea, LotFrontage, TotalBsmtSF, Id))
model_LotF = lm(SalePrice ~ LotFrontage, data = df_train)
summary(model_LotF)
df_train = df_train %>% filter(LotFrontage < 300 | is.na(LotFrontage))
df_train = df_train %>% mutate(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage))
# test_set_LotFrontage = df_train %>% filter(LotFrontage <250)
# model_LotF_test = lm(SalePrice ~ LotFrontage, data = test_set_LotFrontage)
# summary(model_LotF_test)
# Both linear models are significant (p-value 2.2e-16) but R-squared improves when removing 2 outliers
# Outlier IDs --> 935 & 1299
# Can remove outliers, ideally remove ID 1299 as it is also an outlier for other variables

#  ----------------------------------------------------------------------------------
# Check for 'LotArea' as a predictor
df_train %>% group_by(LotArea) %>% summarise(n())
df_train %>% select(c(SalePrice, LotArea)) %>% ggplot() +geom_point(aes(x=LotArea, y = SalePrice))
df_train %>% select(c(SalePrice, LotArea)) %>% ggplot() +geom_point(aes(x=log(LotArea), y = SalePrice))
df_train %>% filter(LotArea >100000) %>% select(c(SalePrice, LotArea, LotFrontage, TotalBsmtSF, Id))
df_train = df_train %>% filter(LotArea < 200000)
model_LotA = lm(SalePrice ~ LotArea, data = df_train)
summary(model_LotA)
model_Lot_log = lm(SalePrice ~ log(LotArea), data = df_train)
summary(model_Lot_log)
# Both linear models are significant (p-value 2.2e-16) but R^2 improves when taking the log of LotArea
# Take the log of LotArea when running the model

#  ----------------------------------------------------------------------------------
# Check for 'Street' as a predictor
df_train %>% group_by(Street) %>% summarise(n())
df_train %>% select(c(SalePrice, Street)) %>% ggplot() +geom_boxplot(aes(x=Street, y = SalePrice))
model_street = lm(SalePrice ~ Street, data = df_train)
summary(model_street)
# Street is NOT a good predictor (p-value of 0.117) 
# Only 6 houses have gravel road access to the property out of 1460 houses, 
# therefore, the Street column can be dropped

#  ----------------------------------------------------------------------------------
# Check for 'Alley' as a predictor
df_train %>% group_by(Alley) %>% summarise(n())
df_train %>% select(c(SalePrice, Alley)) %>% ggplot() +geom_boxplot(aes(x=Alley, y = SalePrice))
model_Alley = lm(SalePrice ~ Alley , data = df_train)
summary(model_Alley)
# Alley is a good predictor (p-value of 0.285) 
View(df_train %>% filter(Alley == "Grvl") %>% select(c(SalePrice, LotArea, Neighborhood)))  
df_train %>% select(c(Alley, Neighborhood, GarageCars)) %>% filter(Alley == "Grvl") %>%  group_by(Neighborhood) %>% summarise(n(), mean(GarageCars))
df_train %>% select(c(Alley, Neighborhood, GarageCars)) %>% filter(Alley == "Pave") %>%  group_by(Neighborhood) %>% summarise(n(), mean(GarageCars))
df_train %>% select(c(Alley, Neighborhood)) %>% filter(Alley == "Pave" | Alley == "Grvl") %>%  group_by(Alley, Neighborhood) %>% summarise(n())
df_train %>% select(c(Alley, Neighborhood)) %>% filter(is.na(Alley)) %>%  group_by(Neighborhood) %>% summarise(n())
df_train %>% filter(Neighborhood == "Somerst") %>% select(c(Alley,Neighborhood)) %>% summarise(n())
df_train %>% filter(LotArea < 11000) %>% group_by(Alley) %>% summarise(n(), mean(GarageCars))

#  ----------------------------------------------------------------------------------
# Check for 'LotShape' as a predictor
df_train %>% group_by(LotShape) %>% summarise(n())
df_train %>% select(c(SalePrice, LotShape)) %>% ggplot() +geom_boxplot(aes(x=LotShape, y = SalePrice))
# IRR2 & IRR3 had only 51 entries out of 1460 and they had significantly higher mean/median than Reg
# So I consolidated IR2 and IR3 into just IR2 to reduce levels
df_train %>% select(c(SalePrice, LotShape)) %>% 
  transmute(SalePrice, LotShape = ifelse((LotShape == 'IR2' | LotShape == 'IR3'), 'IR', LotShape)) %>% 
  mutate(LotShape = recode(LotShape, "1" = "IR1", "4" = "Reg", IR = "IR2")) %>% 
  ggplot() +geom_boxplot(aes(x=LotShape, y = SalePrice))
model_LotShape = lm(SalePrice ~ LotShape, data = df_train)
summary(model_LotShape)
#       NOTE: did not combine all Irregular lot shapes, but below code can do that
# Consolidated number of factors for LotShape from 4 (IR1, IR2, IR3, Reg) to 2 (Irr, Reg)
df_train = df_train %>%  mutate(LotShape = ifelse((LotShape == 'IR2' | LotShape == 'IR3'), 'IR2', 
                                                  ifelse(LotShape == "IR1", "IR1", "Reg")))
# df_train$LotShape = df_train %>% select(LotShape) %>%
#   transmute(LotShape = ifelse((LotShape == 'Reg' ), 'Reg', 'Irr'))

#  ----------------------------------------------------------------------------------
# LandContour
df_train %>% select(c(SalePrice, LandContour)) %>% ggplot() +geom_boxplot(aes(x=LandContour, y = SalePrice))
df_train %>% group_by(LandContour) %>% summarise(n())
# No changes made

#  ----------------------------------------------------------------------------------
# Utilities
df_train %>% group_by(Utilities) %>% summarise(n())
# only one house has no 'All Public Utilities' and therefore variable is insignificant

#  ----------------------------------------------------------------------------------
# Check for 'LotConfig' as a predictor
df_train %>% group_by(LotConfig) %>% summarise(n())
df_train %>% select(c(SalePrice, LotConfig)) %>% ggplot() +geom_boxplot(aes(x=LotConfig, y = SalePrice))
df_train %>% filter(LotConfig == 'Corner', SalePrice >600000) %>% 
  select(c(SalePrice, LotArea, LotFrontage, TotalBsmtSF, Id))
model_LotConfig = lm(SalePrice ~ LotConfig, data = df_train)
summary(model_LotConfig)
# combine two factors 'FR2' and 'FR3' together into a single 'MultSide' to show multiple sided frontage
# of the property
# can remove from model

#  ----------------------------------------------------------------------------------
# Check for 'LandSlope' as a predictor
df_train %>% group_by(LandSlope) %>% summarise(n())
df_train %>% select(c(SalePrice, LandSlope)) %>% ggplot() +geom_boxplot(aes(x=LandSlope, y = SalePrice))
# medians of each overlap within the interquartile ranges, and the different
# factors don't exhibit variability from one another
model_LandSlope = lm(SalePrice ~ LandSlope, data = df_train)
summary(model_LandSlope)
# df_train$LandSlope = df_train %>% select(LandSlope) %>% 
#   transmute(LandSlope = ifelse((LandSlope == 'Gtl' ), 'Low', 'High'))
df_train = df_train %>%  mutate(LandSlope = ifelse((LandSlope == 'Gtl'), 'Low', 'High'))

# Combine Moderate and Severe together to see if it improves the prediction model
model_LandSlope = lm(SalePrice ~ LandSlope, data = df_train)
summary(model_LandSlope)
# p-value of the model is still high; therefore
# do not use LandSlope as a predictor

#  ----------------------------------------------------------------------------------
# Check for 'Neighborhood' as a predictor
View(df_train %>% group_by(Neighborhood) %>% summarise(n(), median(SalePrice),med_area = median(LotArea), sd(SalePrice)) %>% arrange(desc(med_area)))
df_train %>% select(c(SalePrice, LotArea, Neighborhood, TotalBsmtSF, Id))
df_train %>% select(c(SalePrice, Neighborhood)) %>% 
  mutate(Neighborhood = fct_reorder(Neighborhood, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Neighborhood, y = SalePrice, fill = Neighborhood))
# Use the plot to group every 3 neighborhoods together into 1
# Name the groups something along the lines of 'VLow', 'Low', 'Med', 'High', 'VHigh'
model_Neighborhood = lm(SalePrice ~ Neighborhood, data = df_train)
summary(model_Neighborhood)

#  ----------------------------------------------------------------------------------
# Check for 'Condition1' as a predictor
df_train %>% group_by(Condition1) %>% summarise(n())
df_train %>% select(c(SalePrice, LotArea, Condition1, TotalBsmtSF, Id))
df_train %>% select(c(SalePrice, Condition1)) %>% 
  mutate(Condition1 = fct_reorder(Condition1, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Condition1, y = SalePrice, fill = Condition1))
model_Condition1 = lm(SalePrice ~ Condition1, data = df_train)
summary(model_Condition1)
# most factors are significant; the model is also significant
# group a few levels/factors together based off the plot of the sorted median SalePrice
#  Remove from model

#  ----------------------------------------------------------------------------------
# Check for 'Condition2' as a predictor
df_train %>% group_by(Condition2) %>% summarise(n())
df_train %>% select(c(SalePrice, LotArea, Condition1, Condition2, TotalBsmtSF, Id))
df_train %>% mutate(Condition2 = fct_reorder(Condition2, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Condition2, y = SalePrice, fill = Condition2))
model_Condition2 = lm(SalePrice ~ Condition2, data = df_train)
summary(model_Condition2)
# 99% accounts for Normal condition but it is likely tied to Condition1
# Most likely to remove the column entirely


#  ----------------------------------------------------------------------------------
# Check for 'BldgType' as a predictor
df_train %>% group_by(BldgType) %>% summarise(n())
df_train %>% select(c(SalePrice, LotArea, BldgType, TotalBsmtSF, Id))
df_train %>% select(c(SalePrice, BldgType)) %>% 
  mutate(BldgType = fct_reorder(BldgType, SalePrice, .fun = 'median')) %>% 
  ggplot() + geom_boxplot(aes(x=BldgType, y = SalePrice, fill = BldgType))
Kdf_train %>% select(c(SalePrice, BldgType)) %>% ggplot() +geom_point(aes(x=BldgType, y = SalePrice))
df_train %>% select(c(SalePrice, BldgType)) %>% group_by(BldgType) %>% summarise(avg_SP=mean(SalePrice)) %>% 
  ggplot() +geom_point(aes(x=BldgType, y = avg_SP))
model_BldgType = lm(SalePrice ~ BldgType, data = df_train)
summary(model_BldgType)
# Could potentially group '2FmCon', 'Duplex', and 'Twnhs' into one category with
# '1Fam' as the base model
df_train = df_train %>%  mutate(BldgType = ifelse((BldgType == '2fmCon' | BldgType == 'Duplex' | BldgType == 'Twnhs'), 'Other', 
                                                  ifelse(BldgType == "1Fam", "1Fam", "TwnhsE")))
model_BldgType = lm(SalePrice ~ BldgType, data = df_train)
summary(model_BldgType)


#  ----------------------------------------------------------------------------------
# Check for 'HouseStyle' as a predictor
df_train %>% group_by(HouseStyle) %>% summarise(n())
df_train %>% select(c(SalePrice, HouseStyle)) %>% 
  mutate(HouseStyle = fct_reorder(HouseStyle, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=HouseStyle, y = SalePrice))
model_MSSubClass = lm(SalePrice ~ HouseStyle, data = df_train)
summary(model_MSSubClass)
df_train = df_train %>% mutate(HouseStyle = ifelse(HouseStyle == "1Story", "1Story", 
            ifelse((HouseStyle == "2Story" | HouseStyle == "2.5Fin"), "2+Fin", 
                       ifelse(HouseStyle == "SLvl", "SLvl", "Other"))))
# consider grouping the BldgType and HouseStyle into one category with a few levels
# mainly by the number of floors
table(df_train$BldgType,df_train$HouseStyle)
# Review the table of BldgType & HouseStyle
df_train = df_train %>% filter(SalePrice < 700000)
df_train %>% filter(SalePrice > 700000) %>% select(c(Id, SalePrice, HouseStyle, LotArea, Neighborhood, BldgType))

#  ----------------------------------------------------------------------------------
# Check for 'OverallQual' as a predictor
df_train %>% group_by(OverallQual) %>% summarise(n())
df_train$OverallQual = as.factor(df_train$OverallQual)
table(df_train$OverallQual,df_train$OverallCond)
df_train %>% select(c(SalePrice, OverallQual, OverallCond))   %>% 
  ggplot() + geom_boxplot(aes(x = (OverallQual), y = SalePrice))
model_OverallQual = lm(SalePrice ~  OverallQual, data = df_train)
summary(model_OverallQual)
# OverallQual is a great predictor for Houses
# No need for any change

#  ----------------------------------------------------------------------------------
# Check for 'OverallCond' as a predictor
df_train %>% group_by(OverallCond) %>% summarise(n())
table(df_train$OverallQual,df_train$OverallCond)
# df_train$OverallCond = as.factor(df_train$OverallCond)
df_train %>% select(c(SalePrice, OverallCond))   %>% 
  ggplot() + geom_boxplot(aes(x = as.factor(OverallCond), y = SalePrice))
model_OverallCond = lm(SalePrice ~  OverallCond, data = df_train)
summary(model_OverallCond)
model_OverallCondQual = lm(SalePrice ~  OverallCond + OverallQual, data = df_train)
summary(model_OverallCondQual)
# OverallCond is not a significant factor as it seems to be multicollinear with OverallQual
# Do not use OverallCond in the prediction model

#  ----------------------------------------------------------------------------------
# Check for 'YearBuilt' as a predictor
df_train %>% group_by(YearBuilt) %>% summarise(n())
df_train %>% select(c(SalePrice, YearBuilt))   %>% 
  ggplot() + geom_point(aes(x = as.factor(YearBuilt), y = log(SalePrice)))
df_train %>% select(c(SalePrice, YearBuilt)) %>% group_by(YearBuilt) %>% 
  summarise(avg_sale = mean(SalePrice))%>% 
  ggplot() + geom_point(aes(x = as.factor(YearBuilt), y = avg_sale))
model_YearBuilt = lm(SalePrice ~  YearBuilt, data = df_train)
summary(model_YearBuilt)
model_YearBuilt = lm(SalePrice ~  YearBuilt + YearRemodAdd, data = df_train)
summary(model_YearBuilt)
# YearBuilt is a significant factor
# Keep it in the prediction model
# Bin the YearBuilt to different decades
# 1950; 1951-1960; 1961-1970; 1971-1980; 1981-1990; 1991-2000; 2001-2010
# Pre50; 50s; 60s; 70s; 80s; 90s; 20s; 
df_train = df_train %>% mutate(SoldAfterRemod = (as.numeric(YrSold) - as.numeric(YearRemodAdd)),
                    SoldAfterBuilt = (as.numeric(YrSold) - as.numeric(YearBuilt)),
                    RemodAfterBuilt = (as.numeric(YearRemodAdd) - as.numeric(YearBuilt)))

df_train %>% transmute(SalePrice, YearBuilt, YearRemodAdd, YrSold, 
             SoldAfterRemod = (as.numeric(YrSold) - as.numeric(YearRemodAdd))) %>% 
  ggplot() + geom_point(aes(x=SoldAfterRemod, y=SalePrice))
df_train %>% transmute(SalePrice, YearBuilt, YearRemodAdd, YrSold, 
                       SoldAfterBuilt = (as.numeric(YrSold) - as.numeric(YearBuilt))) %>% 
  ggplot() + geom_point(aes(x=SoldAfterBuilt, y=SalePrice))
df_train %>% transmute(SalePrice, YearBuilt, YearRemodAdd, YrSold, 
                       RemodAfterBuilt = (as.numeric(YearRemodAdd) - as.numeric(YearBuilt))) %>% 
  ggplot() + geom_point(aes(x=RemodAfterBuilt, y=SalePrice))

#  ----------------------------------------------------------------------------------
# Check for 'YearRemodAdd' as a predictor
df_train %>% group_by(YearRemodAdd) %>% summarise(n())
df_train$YearRemodAdd = as.factor(df_train$YearRemodAdd)
levels((df_train$YearRemodAdd))
df_train %>% select(c(SalePrice, YearRemodAdd))   %>% 
  ggplot() + geom_point(aes(x = as.factor(YearRemodAdd), y = SalePrice))
df_train %>% select(c(SalePrice, YearRemodAdd)) %>% group_by(YearRemodAdd) %>% 
  summarise(avg_sale = mean(SalePrice))%>% 
  ggplot() + geom_point(aes(x = as.factor(YearRemodAdd), y = avg_sale))
model_YearRemodAdd = lm(SalePrice ~  YearRemodAdd, data = df_train)
summary(model_YearRemodAdd)
model_YearRemodAdd = lm(SalePrice ~  YearRemodAdd + YearBuilt, data = df_train)
summary(model_YearRemodAdd)
# YearRemodAdd is a significant factor
# Keep it in the prediction model
# Bin the YearRemodAdd to different decades
# 1950; 1951-1960; 1961-1970; 1971-1980; 1981-1990; 1991-2000; 2001-2010
# Pre50; 50s; 60s; 70s; 80s; 90s; 20s; 

#  ----------------------------------------------------------------------------------
# RoofStyle
df_train %>% select(c(SalePrice, RoofStyle)) %>% 
  mutate(RoofStyle = fct_reorder(RoofStyle, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=RoofStyle, y = SalePrice))
df_train %>% group_by(RoofStyle) %>% summarise(n())
# use in the model to be dummified

#  ----------------------------------------------------------------------------------
# RoofMatl
df_train %>% select(c(SalePrice, RoofMatl)) %>% 
  mutate(RoofMatl = fct_reorder(RoofMatl, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=RoofMatl, y = SalePrice))
df_train %>% group_by(RoofMatl) %>% summarise(n())
model_RoofMatl = lm(SalePrice ~ RoofMatl, data = df_train)
summary(model_RoofMatl)
# Even though the model seems significant, the slopes/levels are not significant
# since all but one of the levels (CompShg) have very low frequency observation
# Can remove from the model

#  ----------------------------------------------------------------------------------
# Check for 'Exterior1st' as a predictor
df_train %>% group_by(Exterior1st) %>% summarise(n())
df_train %>% select(c(SalePrice, Exterior1st)) %>% ggplot() +geom_boxplot(aes(x=Exterior1st, y = SalePrice))
df_train %>% select(c(SalePrice, Exterior1st)) %>% 
  mutate(Exterior1st = fct_reorder(Exterior1st, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Exterior1st, y = SalePrice))
model_Exterior1st = lm(SalePrice ~ Exterior1st, data = df_train)
summary(model_Exterior1st)
# the model seems significant with a low p-value

#  ----------------------------------------------------------------------------------
# Check for 'Exterior2nd' as a predictor
df_train %>% group_by(Exterior2nd) %>% summarise(n())
df_train %>% select(c(SalePrice, Exterior2nd)) %>% ggplot() +geom_boxplot(aes(x=Exterior2nd, y = SalePrice))
df_train %>% select(c(SalePrice, Exterior2nd)) %>% 
  mutate(Exterior2nd = fct_reorder(Exterior2nd, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Exterior2nd, y = SalePrice))
model_Exterior2nd = lm(SalePrice ~ Exterior2nd, data = df_train)
summary(model_Exterior2nd)
# the model seems significant with a low p-value yet not all levels are
# Use for initial model but could potentially remove

#  ----------------------------------------------------------------------------------
# MasVnrType
df_train %>% select(c(SalePrice, MasVnrType)) %>% ggplot()+ geom_boxplot(aes(x=MasVnrType, y = SalePrice))
df_train %>% group_by(MasVnrType) %>% summarise(n())
# Change NAs to None
df_train = df_train %>% mutate(MasVnrType = ifelse((is.na(MasVnrType) | MasVnrType == "None"), "None", 
                                                 ifelse(MasVnrType == "BrkCmn", "BrkCmn", 
                                                ifelse(MasVnrType == "BrkFace", "BrkFace","Stone"))))
model_MasVnrType = lm(SalePrice ~ MasVnrType, data = df_train)
summary(model_MasVnrType)
# Model seems to be significant
# Grouped the NAs with None

#  ----------------------------------------------------------------------------------
# MasVnrArea
df_train %>% select(c(SalePrice, MasVnrArea)) %>% 
  ggplot() +geom_point(aes(x=MasVnrArea, y = SalePrice))
# imputing 0.00001 for Masonry Veneer Area Null values
# df_train = df_train %>% mutate(MasVnrArea = ifelse(is.na(MasVnrArea), 0.00001, MasVnrArea))
df_train = df_train %>% mutate(MasVnrArea = ifelse(((is.na(MasVnrArea)| MasVnrArea <=1)), 0.00001, MasVnrArea))
df_train %>% group_by(MasVnrType) %>% summarise(n(), min(MasVnrArea), max(MasVnrArea))
df_train %>% filter(MasVnrArea > 1 , MasVnrType == "None") %>% select(c(SalePrice, MasVnrArea, MasVnrType))
# # Classify the 3 "None" MasVnrType with significant MasVnrArea using logistic regression
# log_reg = glm(MasVnrType ~ SalePrice + MasVnrArea ,family = "binomial", data = df_train)
# newdata = df_train %>% filter(MasVnrArea > 1 , MasVnrType == "None") %>% select(c(SalePrice, MasVnrArea))
# predict(log_reg, newdata, type = "response")
df_train %>% group_by(MasVnrType) %>% summarise(n(),min(MasVnrArea), max(MasVnrArea) )
# Keep as a predictor

#  ----------------------------------------------------------------------------------
# ExterQual
df_train %>% select(c(SalePrice, ExterQual)) %>% 
  ggplot() +geom_boxplot(aes(x=ExterQual, y = SalePrice))
df_train %>% group_by(ExterQual) %>% summarise(n())
model_ExterQual = lm(SalePrice ~ ExterQual, data = df_train)
summary(model_ExterQual)
# A very good predictor for the model

#  ----------------------------------------------------------------------------------
# ExterCond
df_train %>% select(c(SalePrice, ExterCond)) %>% 
  ggplot() +geom_boxplot(aes(x=ExterCond, y = SalePrice))
df_train %>% group_by(ExterCond) %>% summarise(n())
model_ExterCond = lm(SalePrice ~ ExterCond, data = df_train)
summary(model_ExterCond)
# Not a good predictor for the model
# Drop from the model



#  ----------------------------------------------------------------------------------
# Foundation
df_train %>% select(c(SalePrice, Foundation)) %>% 
  mutate(Foundation = fct_reorder(Foundation, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Foundation, y = SalePrice))
df_train %>% group_by(Foundation) %>% summarise(n())
# Bin "Wood", "Stone", and "BrkTil" together as "Other"
df_train = df_train %>% mutate(Foundation = (ifelse(((Foundation == "Wood")|
            (Foundation == "Stone")|(Foundation == "BrkTil")), "Other", 
            ifelse(Foundation == "Slab", "Slab",
                   ifelse(Foundation == "CBlock", "CBlock", "PConc")))))
model_Foundation = lm(SalePrice ~ Foundation, data = df_train)
summary(model_Foundation)
# model achieves low p-value and the predictors are significant
# Keep in the model

#  ----------------------------------------------------------------------------------
# BsmtQual
df_train %>% select(c(SalePrice, BsmtQual)) %>% 
  mutate(BsmtQual = fct_reorder(BsmtQual, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=BsmtQual, y = SalePrice))
df_train %>% group_by(BsmtQual) %>% summarise(n())
model_BsmtQual = lm(SalePrice ~ BsmtQual, data = df_train)
summary(model_BsmtQual)
# Very good predictor
# Definitely to be used in the model
df_train = df_train %>% mutate(BsmtQual = (ifelse(BsmtQual == "Ex", "Ex", 
                          ifelse(BsmtQual == "Gd", "Gd", 
                              ifelse(BsmtQual == "TA", "TA", 
                                     ifelse(BsmtQual == "Fa", "Fa", 
                                          ifelse((is.na(BsmtQual)| (!complete.cases(BsmtQual))), "NoBasement", "Po")))))))

#  ----------------------------------------------------------------------------------
# BsmtCond
df_train %>% select(c(SalePrice, BsmtCond)) %>% 
  mutate(BsmtCond = fct_reorder(BsmtCond, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=BsmtCond, y = SalePrice))
df_train %>% group_by(BsmtCond) %>% summarise(n())
model_BsmtCond = lm(SalePrice ~ BsmtCond , data = df_train)
summary(model_BsmtCond)
# Model has low p-value but R^2 is very low so model doesn't explain the variance well
# This is likely because the "TA" level has an overwhelming frequency of observations
# Could remove from the model

#  ----------------------------------------------------------------------------------
# BsmtExposure
df_train %>% select(c(SalePrice, BsmtExposure)) %>% 
  mutate(BsmtExposure = fct_reorder(BsmtExposure, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=BsmtExposure, y = SalePrice))
df_train %>% group_by(BsmtExposure) %>% summarise(n())
# Change NAs to NB or NA to represent no basement
df_train = df_train %>% mutate(BsmtExposure = (
  ifelse(((BsmtExposure == "No")|(is.na(BsmtExposure))), "NoBasement", 
         ifelse(BsmtExposure == "Mn", "Mn", 
                ifelse(BsmtExposure == "Av", "Av", "Gd")))))

# can also combine it with No for No exposure, but like could skew the SalePrice
df_train  %>% filter(SalePrice >500000) %>% group_by(BsmtExposure)%>% summarise(n())
model_BsmtExposure = lm(SalePrice ~ BsmtExposure , data = df_train)
summary(model_BsmtExposure)
# Model has low p-value but R^2 is a bit small
# Keep in initial model

#  ----------------------------------------------------------------------------------
# BsmtFinType1
df_train %>% select(c(SalePrice, BsmtFinType1)) %>%
  mutate(BsmtFinType1 = fct_reorder(BsmtFinType1, SalePrice, .fun = 'median')) %>%
  ggplot() +geom_boxplot(aes(x=BsmtFinType1, y = SalePrice))
df_train %>% filter(!is.na(BsmtFinType1))%>% group_by(BsmtFinType1)%>% 
  summarise(n(), avg_Price = mean(SalePrice), sd_Price = sd(SalePrice), med_Price = median(SalePrice)) %>% 
  arrange(sd_Price,med_Price, avg_Price)
df_train %>% group_by(BsmtFinType1) %>% summarise(n())
# Group 'Rec', 'BLQ', 'LwQ', 'ALQ' into one category called 'AvQ' as they have roughly 
# the same sd, mean, and median
df_train = df_train %>% mutate(BsmtFinType1 = (ifelse(((BsmtFinType1 == "Rec")|(BsmtFinType1 == "BLQ")|(BsmtFinType1 == "LwQ")|(BsmtFinType1 == "ALQ")), "AvQ", 
                                                      ifelse(BsmtFinType1 == "GLQ", "GLQ", 
                                                             ifelse(BsmtFinType1 == "Unf", "Unf", "NoBasement")))))
df_train = df_train %>% mutate(BsmtFinType1 = (ifelse(is.na(BsmtFinType1), "NoBasement", BsmtFinType1)))
# Change NAs to NoBasement to represent no basement
model_BsmtFinType1 = lm(SalePrice ~ BsmtFinType1 , data = df_train)
summary(model_BsmtFinType1)
# Model has low p-value and predictors are significant
# a very good predictor so keep in model

#  ----------------------------------------------------------------------------------
# BsmtFinType2
df_train %>% select(c(SalePrice, BsmtFinType2)) %>%
  mutate(BsmtFinType2 = fct_reorder(BsmtFinType2, SalePrice, .fun = 'median')) %>%
  ggplot() +geom_boxplot(aes(x=BsmtFinType2, y = SalePrice))
df_train %>% filter(!is.na(BsmtFinType2))%>% group_by(BsmtFinType2)%>% 
  summarise(n(), avg_Price = mean(SalePrice), sd_Price = sd(SalePrice), med_Price = median(SalePrice)) %>% 
  arrange(sd_Price,med_Price, avg_Price)
# Group 'Rec', 'BLQ', 'LwQ', into one category called 'AvQ' as they have roughly 
# the same sd, mean, and median

df_train = df_train %>% mutate(BsmtFinType2 = (ifelse(((BsmtFinType2 == "Rec")|(BsmtFinType2 == "BLQ")|(BsmtFinType2 == "LwQ")), "BLQ", 
                                                      ifelse(((BsmtFinType2 == "GLQ")|(BsmtFinType2 == "ALQ")), "GLQ", 
                                                             ifelse(BsmtFinType2 == "Unf", "Unf", "NoBasement")))))
df_train = df_train %>% mutate(BsmtFinType2 = (ifelse(is.na(BsmtFinType2), "NoBasement", BsmtFinType2)))
df_train %>% group_by(BsmtFinType2) %>% summarise(n())

# Change NAs to NB or NA to represent no basement
model_BsmtFinType2 = lm(SalePrice ~ BsmtFinType2 , data = df_train)
summary(model_BsmtFinType2)
# Model has low p-value and predictors are significant
# a very good predictor so keep in model

#  ----------------------------------------------------------------------------------
# BsmtFinSF1
df_train %>% select(c(SalePrice, BsmtFinSF1)) %>%
  ggplot() + geom_point(aes(x=BsmtFinSF1, y = SalePrice))
df_train %>% filter(BsmtFinSF1 < 1) %>% select(c(SalePrice, BsmtFinSF1, Id, LotArea, BsmtFinType1))
# remove outlier with high Bsmt FinSF1, Id --> 1299
# df_train$BsmtFinSF1 = df_train %>% filter(BsmtFinSF1 < 4000) %>% select(BsmtFinSF1)
model_BsmtFinSF1 = lm(SalePrice ~ (BsmtFinSF1) , data = df_train)
summary(model_BsmtFinSF1)
# Model has low p-value, with low R^2 but good predictor p-value; keep in model

#  ----------------------------------------------------------------------------------
# BsmtUnfSF
df_train %>% select(c(SalePrice, BsmtUnfSF)) %>%
  ggplot() + geom_point(aes(x=(BsmtUnfSF), y = SalePrice))
df_train %>% filter(BsmtUnfSF <0.1) %>% summarise(n())
# 188 zero values
# or can take log(BsmtFinSF2) of the variable without removing outlier
# df_train$BsmtFinSF2 = df_train %>% filter(BsmtFinSF2 < 1400) %>% select(BsmtFinSF2)
model_BsmtFinSF2 = lm(SalePrice ~ (BsmtFinSF2) , data = df_train)
summary(model_BsmtFinSF2)
# Model has high p-value, so not a good predictor
# do not inlude in final model

#  ----------------------------------------------------------------------------------
# BsmtFinSF2
df_train %>% select(c(SalePrice, BsmtFinSF2)) %>%
  ggplot() + geom_point(aes(x=log(BsmtFinSF2), y = SalePrice))
df_train %>% filter(!is.na(BsmtFinSF2))
df_train %>% filter(BsmtFinSF2 > 1400) %>% select(c(SalePrice, BsmtFinSF2, Id))
# remove outlier with high BsmtFinSF2, Id --> 323
# or can take log(BsmtFinSF2) of the variable without removing outlier
# df_train$BsmtFinSF2 = df_train %>% filter(BsmtFinSF2 < 1400) %>% select(BsmtFinSF2)

model_BsmtFinSF2 = lm(SalePrice ~ (BsmtFinSF2) , data = df_train)
summary(model_BsmtFinSF2)
# Model has high p-value, so not a good predictor
# do not inlude in final model

# Inputing BsmtFinSF2 for Unfinished where BsmtFinSF1 is not Unf. 
df_train$BsmtFinSF2 = ifelse(((df_train$BsmtFinSF2 == 0) & (df_train$BsmtFinSF1 > 0)), 
                             df_train$BsmtUnfSF, df_train$BsmtFinSF2)
# INPUTING THE 0 VALUES FOR UNFINISHED Basement SF 1 & 2 using the BsmtUnfSF variable
ratio_SF1 = df_train %>% summarise(sum(BsmtFinSF1)/ (sum(BsmtFinSF1)+sum(BsmtFinSF2)))
ratio_SF2 = df_train %>% summarise(sum(BsmtFinSF2)/ (sum(BsmtFinSF1)+sum(BsmtFinSF2)))
# imputing the missing values of SF1 and SF2 where basement is Unfinished by using the ratio total SF1 & SF2
# and multiplying it by the TotalBsmtSF to compute each of SF1 and SF2 per the mean distribution
# for (i in (1:nrow(df_train %>% 
#                   filter((df_train$BsmtFinType1 == 'Unf') & (df_train$BsmtFinType2 == 'Unf'))))){
#   df_train$BsmtFinSF1[i] = ifelse( ((df_train$BsmtFinSF1[i] == 0)&
#                     (df_train$BsmtFinSF2[i] == 0)&
#                     (df_train$BsmtFinType1[i] == 'Unf')), 
#                     round(ratio_SF1*df_train$BsmtUnfSF[i]), df_train$BsmtFinSF1[i])
#   df_train$BsmtFinSF2[i] = ifelse( ((df_train$BsmtFinSF2[i] == 0)&
#                     (df_train$BsmtFinType1[i] == 'Unf')), 
#                     round(ratio_SF2*df_train$TotalBsmtSF[i]), 
#                     df_train$BsmtFinSF2[i])
# }
for (i in (1:nrow(df_train))){
  df_train$BsmtFinSF1[i] = ifelse( ((df_train$BsmtFinSF1[i] == 0)&
                                      (df_train$BsmtFinSF2[i] == 0)&
                                      (df_train$BsmtFinType1[i] == 'Unf')), 
                                   round(ratio_SF1*df_train$BsmtUnfSF[i]), df_train$BsmtFinSF1[i])
  df_train$BsmtFinSF2[i] = ifelse( ((df_train$BsmtFinSF2[i] == 0)&
                                      (df_train$BsmtFinType1[i] == 'Unf')), 
                                   round(ratio_SF2*df_train$TotalBsmtSF[i]), 
                                   df_train$BsmtFinSF2[i])
}
# unlist BsmtFinSF1 & BsmtFinSF2 and converting them to numeric
df_train$BsmtFinSF1 = as.numeric(unlist(df_train$BsmtFinSF1))
df_train$BsmtFinSF2 = as.numeric(unlist(df_train$BsmtFinSF2))
# df_train = df_train %>% mutate(BsmtFinSF = (as.numeric(unlist(BsmtFinSF1))+as.numeric(unlist(BsmtFinSF2))))
# built test dataset for SF1, SF2, TotalBsmtSF, BsmtUnfSF,YearBuilt, GarageArea, LotArea, SalePrice
test_model_TotalSF = df_train %>% 
  transmute(BsmtFinSF = (as.numeric(unlist(BsmtFinSF1))+as.numeric(unlist(BsmtFinSF2))), 
            TotalBsmtSF, BsmtUnfSF,YearBuilt, GarageArea, LotArea, SalePrice)

model.saturated = lm(SalePrice~. , data = test_model)
model1 = lm(SalePrice ~ . - BsmtFinSF1, data = test_model)
model2 = lm(SalePrice ~ . - BsmtFinSF2, data = test_model)
model.empty = lm(SalePrice ~ 1 , data = test_model)
model.FinSF = lm(SalePrice ~ TotalBsmtSF + YearBuilt + LotArea, data=test_model_TotalSF)
model.BsmtFinSF = lm(SalePrice ~ TotalBsmtSF + YearBuilt + LotArea + BsmtFinSF, data=test_model_TotalSF)

summary(model.saturated)    # BsmtFinSF2 is not a significant factor
summary(model1)             # BsmtFinSF2 & BsmtUnfSF are not significant factors
summary(model2)
summary(model.empty)
summary(model.FinSF)
summary(model.BsmtFinSF)    # BsmtFinSF is not a significant factor
# the multi-linear models show that BsmtFinSF2 & BsmtUnfSF & BsmtFinSF
# Remove BsmtFinSF2 & BsmtUnfSF from the final model
# TotalBsmtSF & BsmtFinSF1 are very significant for the model

#  ----------------------------------------------------------------------------------
# Check for 'Heating' as a predictor
df_train %>% group_by(Heating) %>% summarise(n())
# very low frequency for all levels apart from GasA
df_train %>% select(c(SalePrice, Heating)) %>% ggplot() +geom_boxplot(aes(x=Heating, y = SalePrice))
model_Heating = lm(SalePrice ~ Heating, data = df_train)
summary(model_Heating)
# model has low p-value but coefficients are not significant
# can remove from the model

#  ----------------------------------------------------------------------------------
# Check for 'HeatingQC' as a predictor
df_train %>% group_by(HeatingQC) %>% summarise(n())
df_train %>% select(c(SalePrice, HeatingQC)) %>% 
  mutate(HeatingQC = fct_reorder(HeatingQC, SalePrice, .fun = 'median')) %>% 
ggplot() +geom_boxplot(aes(x=HeatingQC, y = SalePrice))
# Binned levels together with similar std and/or median
df_train = df_train %>% mutate(HeatingQC = (ifelse(((HeatingQC == "Fa")|(HeatingQC == "Po")), "Bad", 
                                                      ifelse(((HeatingQC == "TA")|(HeatingQC == "Gd")), "Good", "Excellent"))))
model_HeatingQC = lm(SalePrice ~ HeatingQC, data = df_train)
summary(model_HeatingQC)
# model has low p-value and coefficients are significant, apart from HeatingQC 'Poor'
# Keep in the model

#  ----------------------------------------------------------------------------------
# Check for 'CentralAir' as a predictor
df_train %>% group_by(CentralAir) %>% summarise(n())
df_train %>% select(c(SalePrice, CentralAir)) %>% 
  mutate(CentralAir = fct_reorder(CentralAir, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=CentralAir, y = SalePrice))
model_CentralAir = lm(SalePrice ~ CentralAir, data = df_train)
summary(model_CentralAir)
# model has low p-value and coefficients are significant, apart from CentralAir 'Poor'
# Keep in the model

#  ----------------------------------------------------------------------------------
# Check for 'Electrical' as a predictor
df_train %>% group_by(Electrical) %>% summarise(n(), mean(SalePrice),sd(SalePrice),median(SalePrice))
# 91% of observations are SBrkr
df_train %>% select(c(SalePrice, Electrical)) %>% 
  mutate(CentralAir = fct_reorder(Electrical, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=Electrical, y = SalePrice))
# impute the missing NAs using K-nearest neighbour
df_train$Electrical = kNN(df_train %>% select(c(SalePrice, Electrical)), k=5)$Electrical
# Bin 'FuseF' and 'FuseP' together as 'FuseB' for Bad Fuse
df_train = df_train %>% mutate(Electrical = ifelse((Electrical == "FuseF"| Electrical == "FuseP" | Electrical == "Mix"), "FuseB", 
                                                   ifelse(Electrical == "FuseA","FuseG", "SBrkr")))
model_Electrical = lm(SalePrice ~ Electrical, data = df_train)
summary(model_Electrical)
# model has low p-value butlow R^2 so model does not explain the variance well
# Can keep in model

#  ----------------------------------------------------------------------------------
# Check for 'X1stFlrSF' as a predictor
df_train %>% group_by(X1stFlrSF) %>% summarise(n())
df_train %>% select(c(SalePrice, X1stFlrSF)) %>% 
  ggplot() +geom_point(aes(x=X1stFlrSF, y = SalePrice))
df_train %>% filter(X1stFlrSF <4000) %>% select(c(Id, SalePrice, X1stFlrSF)) %>% 
  ggplot() +geom_point(aes(x=X1stFlrSF, y = SalePrice))
# Remove outlier with X1stFlrSF >4000; Id is 1299
df_train = df_train %>% filter(X1stFlrSF <4000)
model_X1stFlrSF = lm(SalePrice ~ X1stFlrSF, data = df_train)
summary(model_X1stFlrSF)
# model has low p-value and coefficients are  significant
# Remove outlier and use X1stFlrSF variable in model

#  ----------------------------------------------------------------------------------
# Check for 'X2ndFlrSF' as a predictor
df_train %>% select(X2ndFlrSF, SalePrice) %>% group_by(X2ndFlrSF) %>% summarise(n())
df_train %>% select(X2ndFlrSF, SalePrice) %>% group_by(X2ndFlrSF) %>% summary()
df_train %>% select(c(SalePrice, X2ndFlrSF)) %>% 
  ggplot() +geom_point(aes(x=X2ndFlrSF, y = SalePrice))
# Create binary variable to determine whether there is a 2nd floor or not
# X2ndFlrYN: 2nd Floor --> No = There is no 2nd floor ; Yes = There is a 2nd floor
for (i in (1:nrow(df_train))){
  df_train$X2ndFlrYN[i] = (ifelse(df_train$X2ndFlrSF[i] == 0, "No", "Yes"))
}
df_train$X2ndFlrYN = as.factor(df_train$X2ndFlrYN)
model_X2ndFlrSF = lm(SalePrice ~ X2ndFlrSF, data = df_train)
summary(model_X2ndFlrSF)
model_X2ndFlrYN = lm(SalePrice ~ X2ndFlrYN, data = df_train)
summary(model_X2ndFlrYN)
model_Floor2= lm(SalePrice ~ X2ndFlrYN + X2ndFlrSF, data = df_train)
summary(model_Floor2)
# X2ndFlrYN variable is not as good a predictor as X2ndFlrSF

#  ----------------------------------------------------------------------------------
# Check for 'LowQualFinSF' as a predictor
df_train %>% select(c(SalePrice, LowQualFinSF)) %>% ggplot() +geom_point(aes(x=LowQualFinSF, y = SalePrice))
model_LowQualFinSF = lm(SalePrice ~ LowQualFinSF, data = df_train)
summary(model_LowQualFinSF)
# LowQualFinSF is not a good predictor
# Remove from the model

#  ----------------------------------------------------------------------------------
# Check for 'GrLivArea' as a predictor
df_train %>% select(c(SalePrice, GrLivArea)) %>% ggplot() +geom_point(aes(x=GrLivArea, y = SalePrice))
df_train %>% filter(GrLivArea>4500) %>%  select(c(SalePrice, LotArea, GrLivArea, Id))
model_GrLivArea = lm(SalePrice ~ GrLivArea, data = df_train)
summary(model_GrLivArea)
# GrLivArea is a very good predictor
# Keep in the model

#  ----------------------------------------------------------------------------------
# Check for 'BsmtFullBath' as a predictor
df_train = df_train %>% filter(BsmtFullBath < 3)
# df_train$BsmtFullBath = as.factor(df_train$BsmtFullBath)   # Convert BsmtFullBath to a factor from integer
df_train %>% group_by(BsmtFullBath) %>% summarise(n())
# df_train$BsmtFullBath = ifelse((df_train$BsmtFullBath == "2") , 
#                         "2", ifelse(df_train$BsmtFullBath == "1", "1", "0"))
df_train %>% select(c(SalePrice, BsmtFullBath)) %>% 
  ggplot() +geom_boxplot(aes(x=BsmtFullBath, y = SalePrice))
model_BsmtFullBath = lm(SalePrice ~ BsmtFullBath, data = df_train)
summary(model_BsmtFullBath)
# BsmtFullBath is a good predict, but has little variation of SalePrice for different BsmtFullBath levels
# Can keep in model

#  ----------------------------------------------------------------------------------
# Check for 'BsmtHalfBath' as a predictor
df_train$BsmtHalfBath = as.factor(df_train$BsmtHalfBath)   # Convert BsmtFullBath to a factor from integer
df_train %>% group_by(BsmtHalfBath) %>% summarise(n())
df_train %>% select(c(SalePrice, BsmtHalfBath)) %>% 
  ggplot() +geom_boxplot(aes(x=BsmtHalfBath, y = SalePrice))
df_train %>% group_by(BsmtHalfBath) %>% summarise(n())
model_BsmtHalfBath = lm(SalePrice ~ BsmtHalfBath, data = df_train)
summary(model_BsmtHalfBath)
# BsmtHalfBath not a good predictor
# Remove from model

#  ----------------------------------------------------------------------------------
# Check for 'FullBath' as a predictor
df_train$FullBath = as.factor(df_train$FullBath)   # Convert BsmtFullBath to a factor from integer
df_train %>% select(c(SalePrice, FullBath)) %>% 
  ggplot() +geom_boxplot(aes(x=FullBath, y = SalePrice))
df_train %>% group_by(FullBath) %>% summarise(n())
df_train %>% filter(FullBath == 0) %>% select(c(FullBath, Id,BsmtFullBath,BsmtHalfBath,BedroomAbvGr, SalePrice, LotArea ))
model_FullBath = lm(SalePrice ~ FullBath, data = df_train)
summary(model_FullBath)
df_train %>% filter(FullBath == '0') %>% 
  select(c(SalePrice, HalfBath, FullBath, BsmtFullBath, BsmtHalfBath ,Id))
# Remove the houses without any baths?
df_train_fullbath = df_train %>% filter(FullBath != '0')
model_HasFullBath = lm(SalePrice ~ FullBath, data = df_train_fullbath)
summary(model_HasFullBath)
# FullBath is a good predictor once removing the outliers with no full baths

df_train$TotalBath = (df_train$FullBath + df_train$BsmtFullBath + 0.5*df_train$BsmtHalfBath + 0.5*df_train$HalfBath)
df_train %>% group_by(TotalBath) %>% summarise(n())

#  ----------------------------------------------------------------------------------
# Check for 'HalfBath' as a predictor
df_train$HalfBath = as.factor(df_train$HalfBath)   # Convert BsmtFullBath to a factor from integer
df_train %>% select(c(SalePrice, HalfBath)) %>% 
  ggplot() +geom_boxplot(aes(x=HalfBath, y = SalePrice))
df_train %>% group_by(HalfBath) %>% summarise(n())
model_HalfBath = lm(SalePrice ~ HalfBath, data = df_train)
summary(model_HalfBath)
# Create binary variable to determine whether there is a half bath or not
# HalfBath: Half Bath--> No = There is no half bath ; Yes = There is a half bath
for (i in (1:nrow(df_train))){
  df_train$HalfBathYN[i] = (ifelse(df_train$HalfBath[i] == 0, "No", "Yes"))
}
df_train$HalfBathYN = as.factor(df_train$HalfBathYN)
model_HalfBathYN = lm(SalePrice ~ HalfBathYN, data = df_train)
summary(model_HalfBathYN)
# FullBath is a good predictor; can use binary HalfBath 'HalfBathYN' or numeric 'HalfBath'
# Remove from model

#  ----------------------------------------------------------------------------------
# Check for 'BedroomAbvGr' as a predictor
df_train$BedroomAbvGr = as.factor(df_train$BedroomAbvGr)   # Convert BsmtFullBath to a factor from integer
df_train %>% select(c(SalePrice, BedroomAbvGr)) %>% 
  ggplot() +geom_boxplot(aes(x=BedroomAbvGr, y = SalePrice))
df_train %>% group_by(BedroomAbvGr) %>% summarise(n())
df_train %>% filter(BedroomAbvGr ==0 ) %>% 
  select(c(SalePrice, GrLivArea, BedroomAbvGr, FullBath, Id))
model_BedroomAbvGr = lm(SalePrice ~ BedroomAbvGr, data = df_train)
summary(model_BedroomAbvGr)
# Bedroom is not a good predictor, even after removing the outliers

#  ----------------------------------------------------------------------------------
# Check for 'KitchenAbvGr' as a predictor
class(df_train$KitchenAbvGr)
df_train$KitchenAbvGr = as.factor(df_train$KitchenAbvGr)
df_train %>% select(c(SalePrice, KitchenAbvGr)) %>% 
  ggplot() +geom_boxplot(aes(x=KitchenAbvGr, y = SalePrice))
df_train %>% group_by(KitchenAbvGr) %>% summarise(n())
# KitchenAbvGr is not a good predictor, as most observations (95%) have 1 kitchen


#  ----------------------------------------------------------------------------------
# Check for 'KitchenQual' as a predictor
df_train %>% select(c(SalePrice, KitchenQual)) %>% 
  ggplot() +geom_boxplot(aes(x=KitchenQual, y = SalePrice))
df_train %>% group_by(KitchenQual) %>% summarise(n())
model_KitchenQual = lm(SalePrice ~ KitchenQual, data = df_train)
summary(model_KitchenQual)
# KitchenQual is a very good predictor
# Add to model

#  ----------------------------------------------------------------------------------
# Check for 'TotRmsAbvGrd' as a predictor
class(df_train$TotRmsAbvGrd)
# df_train$TotRmsAbvGrd = as.factor(df_train$TotRmsAbvGrd)
df_train %>% select(c(SalePrice, TotRmsAbvGrd)) %>% 
  ggplot() +geom_boxplot(aes(x=TotRmsAbvGrd, y = SalePrice))
df_train %>% filter(TotRmsAbvGrd == '14')  %>% 
  select(c(SalePrice, LotArea, TotRmsAbvGrd, FullBath, Id))
df_train = df_train %>% filter(TotRmsAbvGrd <= 13)
# could potentially remove this outlier with 14 rooms above grade; Id 636
df_train %>% group_by(TotRmsAbvGrd) %>% summarise(n())
model_TotRmsAbvGrd = lm(SalePrice ~ TotRmsAbvGrd, data = df_train)
summary(model_TotRmsAbvGrd)
# TotRmsAbvGrd is a good predictor but some levels are not significant
# Add to model (?)

#  ----------------------------------------------------------------------------------
# Check for 'Functional' as a predictor
df_train %>% select(c(SalePrice, Functional)) %>% 
  ggplot() +geom_boxplot(aes(x=Functional, y = SalePrice))
df_train %>% group_by(Functional) %>% summarise(n())

df_train = df_train %>% mutate(Functional = ifelse((Functional == "Typ"), "Typ", "Other"))

model_TotRmsAbvGrd = lm(SalePrice ~ Functional, data = df_train)
summary(model_TotRmsAbvGrd)
# Functional linear model has a low p-value and low R^2
# Can keep for initial model



#  ----------------------------------------------------------------------------------
# Check for 'Fireplaces' as a predictor
df_train$Fireplaces = as.factor(df_train$Fireplaces)
df_train %>% select(c(SalePrice, Fireplaces)) %>% 
  ggplot() +geom_boxplot(aes(x=Fireplaces, y = SalePrice))
df_train %>% group_by(Fireplaces) %>% summarise(n())

model_Fireplaces = lm(SalePrice ~ Fireplaces, data = df_train)
summary(model_Fireplaces)
# Functional linear model has a low p-value and good R^2
# Good predictor to be used

#  ----------------------------------------------------------------------------------
# Check for 'FireplaceQu' as a predictor
df_train %>% select(c(SalePrice, FireplaceQu)) %>% 
  ggplot() +geom_boxplot(aes(x=FireplaceQu, y = SalePrice))
df_train %>% group_by(FireplaceQu) %>% summarise(n())

df_train = df_train %>% mutate(FireplaceQu = ifelse(is.na(FireplaceQu), "None", 
          ifelse(FireplaceQu == "Ex", "Ex",ifelse(FireplaceQu == "Fa", "Fa",
          ifelse(FireplaceQu == "Gd", "Gd", ifelse(FireplaceQu == "Po", "Po", "TA"))))))


model_FireplaceQu = lm(SalePrice ~ FireplaceQu, data = df_train)
summary(model_FireplaceQu)
# FireplaceQu linear model has a low p-value and good R^2
# Good predictor to be used


#  ----------------------------------------------------------------------------------
# Check for 'GarageType' as a predictor
df_train %>% select(c(SalePrice, GarageType)) %>% 
  ggplot() +geom_boxplot(aes(x=GarageType, y = SalePrice))
df_train %>% group_by(GarageType) %>% summarise(n())

df_train = df_train %>% mutate(GarageType = ifelse(is.na(GarageType), "No Garage", 
                          ifelse(GarageType == "2Types", "2Types",ifelse(GarageType == "Attchd", "Attchd",
                          ifelse(GarageType == "Basment", "Basment", ifelse(GarageType == "BuiltIn", "BuiltIn", 
                          ifelse(GarageType == "CarPort", "CarPort", "Detchd")))))))

model_GarageType = lm(SalePrice ~ GarageType, data = df_train)
summary(model_GarageType)
# GarageType linear model has a low p-value and good R^2
# Can use in first model

#  ----------------------------------------------------------------------------------
# Check for 'GarageYrBlt' as a predictor
df_train %>% select(c(SalePrice, GarageYrBlt)) %>% 
  ggplot() +geom_point(aes(x=GarageYrBlt, y = SalePrice))
df_train %>% group_by(GarageYrBlt) %>% summarise(n())
df_train %>% select(c(YearBuilt, GarageYrBlt, Id, SalePrice))
df_train %>% transmute(diff_years = (GarageYrBlt - YearBuilt)) %>% ggplot() +geom_histogram(aes(x=diff_years))
# shows that the frequency of Garages built long after the house was built is very low
# could result in multicollinearity
model_GarageYrBlt = lm(SalePrice ~ GarageYrBlt , data = df_train)
summary(model_GarageYrBlt)
# Even thoughh GarageYrBlt linear model has a low p-value and good R^2
# There is a correlation with YearBuilt
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'GarageFinish' as a predictor
df_train %>% select(c(SalePrice, GarageFinish)) %>% 
  ggplot() +geom_boxplot(aes(x=GarageFinish, y = SalePrice))
df_train %>% group_by(GarageFinish) %>% summarise(n())
df_train = df_train %>% mutate(GarageFinish = ifelse(is.na(GarageFinish), "No Garage", 
                                                   ifelse(GarageFinish == "Fin", "Fin",
                                                   ifelse(GarageFinish == "RFn", "RFn", "Unf"))))
model_GarageFinish = lm(SalePrice ~ GarageFinish , data = df_train)
summary(model_GarageFinish)
# GarageFinish linear model has a low p-value and good R^2
# Include in the model

#  ----------------------------------------------------------------------------------
# Check for 'GarageCars' as a predictor
df_train$GarageCars = as.factor(df_train$GarageCars)
df_train %>% select(c(SalePrice, GarageCars)) %>% 
  ggplot() +geom_boxplot(aes(x=GarageCars, y = SalePrice))
df_train %>% group_by(GarageCars) %>% summarise(n())

model_GarageCars = lm(SalePrice ~ GarageCars , data = df_train)
summary(model_GarageCars)
# GarageCars linear model has a low p-value and good R^2
# Include in the model

#  ----------------------------------------------------------------------------------
# Check for 'GarageArea' as a predictor
df_train$GarageArea = as.factor(df_train$GarageArea)
df_train %>% select(c(SalePrice, GarageArea)) %>% 
  ggplot() +geom_point(aes(x=GarageArea, y = SalePrice))
model_GarageArea = lm(SalePrice ~ GarageArea , data = df_train)
summary(model_GarageArea)
# GarageArea linear model has a low p-value and good R^2
# Include in the model
# There could be multicollinearity with GarageCars

#  ----------------------------------------------------------------------------------
# Check for 'GarageQual' as a predictor

df_train %>% select(c(SalePrice, GarageQual)) %>% 
  ggplot() +geom_boxplot(aes(x=GarageQual, y = SalePrice))
df_train %>% group_by(GarageQual) %>% summarise(n())

df_train = df_train %>% mutate(GarageQual = ifelse(is.na(GarageQual), "No Garage", 
              ifelse(GarageQual == "Ex", "Ex",ifelse(GarageQual == "Fa", "Fa",
              ifelse(GarageQual == "Gd", "Gd", ifelse(GarageQual == "Po", "Po", "TA"))))))

model_GarageQual = lm(SalePrice ~ GarageQual , data = df_train)
summary(model_GarageQual)
# GarageCars linear model has a low p-value but R^2 and factors p-value are not significant
# Do not include in the model


#  ----------------------------------------------------------------------------------
# Check for 'GarageCond' as a predictor
df_train$GarageCond = as.factor(df_train$GarageCond)
df_train %>% select(c(SalePrice, GarageCond)) %>% 
  ggplot() +geom_boxplot(aes(x=GarageCond, y = SalePrice))
df_train %>% group_by(GarageCond) %>% summarise(n())

df_train = df_train %>% mutate(GarageCond = ifelse(is.na(GarageCond), "No Garage", 
               ifelse(GarageCond == "Ex", "Ex",ifelse(GarageCond == "Fa", "Fa",
               ifelse(GarageCond == "Gd", "Gd", ifelse(GarageCond == "Po", "Po", "TA"))))))

model_GarageCond = lm(SalePrice ~ GarageCond , data = df_train)
summary(model_GarageCond)
# GarageCond linear model has a low p-value but R^2 and factors p-value are not significant
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'PavedDrive' as a predictor
df_train %>% select(c(SalePrice, PavedDrive)) %>% 
  ggplot() +geom_boxplot(aes(x=PavedDrive, y = SalePrice))
df_train %>% group_by(PavedDrive) %>% summarise(n())

df_train = df_train %>% mutate(PavedDrive = ifelse(PavedDrive == "Y", "Paved", "PartialNo"))

model_PavedDrive = lm(SalePrice ~ PavedDrive , data = df_train)
summary(model_PavedDrive)
# PavedDrive linear model has a low p-value,  R^2 and factors p-value are somewhat significant
#Include in the initial model

#  ----------------------------------------------------------------------------------
# Check for 'WoodDeckSF' as a predictor
df_train$WoodDeckSF = as.factor(df_train$WoodDeckSF)
df_train %>% select(c(SalePrice, WoodDeckSF)) %>% 
  ggplot() +geom_point(aes(x=WoodDeckSF, y = SalePrice))
df_train %>% group_by(WoodDeckSF) %>% summarise(n())

df_train = df_train %>% mutate(WoodDeckSF = ifelse(WoodDeckSF == 0, 0.01, WoodDeckSF))

model_PavedDrive = lm(SalePrice ~ sqrt(WoodDeckSF) , data = df_train)
summary(model_PavedDrive)
# WoodDeckSF linear model has a low p-value,  R^2 and factors p-value are somewhat significant
# Include in the initial model

#  ----------------------------------------------------------------------------------
# Check for 'OpenPorchSF' as a predictor
df_train$OpenPorchSF = as.factor(df_train$OpenPorchSF)
df_train %>% select(c(SalePrice, OpenPorchSF)) %>% 
  ggplot() +geom_point(aes(x=OpenPorchSF, y = SalePrice))
df_train %>% group_by(OpenPorchSF) %>% summarise(n())
df_train = df_train %>% mutate(OpenPorchSF = ifelse(OpenPorchSF == 0, 0.01, OpenPorchSF))
model_OpenPorchSF = lm(SalePrice ~ log(OpenPorchSF) , data = df_train)
summary(model_OpenPorchSF)
# OpenPorchSF linear model has a low p-value,  R^2 and factors p-value are somewhat significant
# Include in the initial model

#  ----------------------------------------------------------------------------------
# Check for 'EnclosedPorch' as a predictor

df_train %>% select(c(SalePrice, EnclosedPorch)) %>% 
  ggplot() +geom_point(aes(x=EnclosedPorch, y = SalePrice))
df_train %>% group_by(EnclosedPorch) %>% summarise(n())
df_train = df_train %>% mutate(EnclosedPorch = ifelse(EnclosedPorch == 0, 0.01, EnclosedPorch))
model_EnclosedPorch = lm(SalePrice ~ log(EnclosedPorch) , data = df_train)
summary(model_EnclosedPorch)
# EnclosedPorch linear model has a low p-value,  R^2 and factors p-value are somewhat significant
# Include in the initial model

#  ----------------------------------------------------------------------------------
# Check for 'X3SsnPorch' as a predictor
df_train %>% select(c(SalePrice, X3SsnPorch)) %>% 
  ggplot() +geom_point(aes(x=X3SsnPorch, y = SalePrice))
df_train %>% group_by(X3SsnPorch) %>% summarise(n())
model_X3SsnPorch = lm(SalePrice ~ (X3SsnPorch) , data = df_train)
summary(model_X3SsnPorch)
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'ScreenPorch' as a predictor
df_train %>% select(c(SalePrice, ScreenPorch)) %>% 
  ggplot() +geom_point(aes(x=ScreenPorch, y = SalePrice))
df_train %>% group_by(ScreenPorch) %>% summarise(n())
df_train = df_train %>% mutate(ScreenPorch = ifelse(ScreenPorch == 0, 0.01, ScreenPorch))
model_ScreenPorch = lm(SalePrice ~ I(ScreenPorch^2) , data = df_train)
summary(model_ScreenPorch)
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'PoolArea' as a predictor
df_train %>% select(c(SalePrice, PoolArea)) %>% 
  ggplot() +geom_point(aes(x=PoolArea, y = SalePrice))
df_train %>% group_by(PoolArea) %>% summarise(n())
df_train = df_train %>% mutate(PoolArea = ifelse(PoolArea == 0, 0.01, PoolArea))
model_PoolArea = lm(SalePrice ~ PoolArea , data = df_train)
summary(model_PoolArea)
# Do not include in the model
# very few observations with pool access

#  ----------------------------------------------------------------------------------
# Check for 'PoolQC' as a predictor
df_train %>% select(c(SalePrice, PoolQC)) %>% 
  ggplot() +geom_point(aes(x=PoolQC, y = SalePrice))
df_train %>% group_by(PoolQC) %>% summarise(n())
df_train = df_train %>% mutate(PoolQC = ifelse(PoolQC == 0, 0.01, PoolQC))
model_PoolArea = lm(SalePrice ~ PoolQC , data = df_train)
summary(model_PoolArea)
# Do not include in the model
# very few observations with pool access

#  ----------------------------------------------------------------------------------
# Check for 'Fence' as a predictor
df_train %>% select(c(SalePrice, Fence)) %>% 
  ggplot() +geom_boxplot(aes(x=Fence, y = SalePrice))
df_train %>% group_by(Fence) %>% summarise(n())
df_train = df_train %>% mutate(Fence = ifelse(is.na(Fence), "No Fence",
            ifelse(Fence == "GdPrv", "GdPrv",ifelse(Fence == "GdWo", "GdWo",
            ifelse(Fence == "MnPrv", "MnPrv", "MnWw")))))

model_Fence = lm(SalePrice ~ Fence , data = df_train)
summary(model_Fence)
# Remove from model

#  ----------------------------------------------------------------------------------
# Check for 'MiscFeature' as a predictor
df_train %>% select(c(SalePrice, MiscFeature)) %>% 
  ggplot() +geom_boxplot(aes(x=MiscFeature, y = SalePrice))
model_Fence = lm(SalePrice ~ MiscFeature , data = df_train)
summary(model_Fence)
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'MiscVal' as a predictor
df_train %>% select(c(SalePrice, MiscVal)) %>% 
  ggplot() +geom_point(aes(x=MiscVal, y = SalePrice))
df_train %>% group_by(MiscVal) %>% summarise(n())
model_MiscVal = lm(SalePrice ~ MiscVal , data = df_train)
summary(model_MiscVal)
# Do not include in the model
# very few observations with MiscVal

#  ----------------------------------------------------------------------------------
# Check for 'MoSold' as a predictor
df_train$MoSold = as.factor(df_train$MoSold)
df_train %>% select(c(SalePrice, MoSold)) %>% 
  ggplot() +geom_boxplot(aes(x=MoSold, y = SalePrice))
df_train %>% group_by(MoSold) %>% summarise(n())
model_MoSold = lm(SalePrice ~ MoSold , data = df_train)
summary(model_MoSold)
# No significance
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'YrSold' as a predictor
df_train$YrSold = as.factor(df_train$YrSold)
df_train %>% select(c(SalePrice, YrSold)) %>% 
  ggplot() +geom_boxplot(aes(x=YrSold, y = SalePrice))
df_train %>% group_by(YrSold) %>% summarise(n())
bbb = df_train %>% group_by(YrSold) %>% summarise(n(), price_persqft = mean(SalePrice)/mean(LotArea))  
bbb %>%   ggplot() + geom_line(aes(x = YrSold, y = price_persqft))

df_train %>% transmute(YrSold, SalePrice, CostPerSqft = (SalePrice/(X1stFlrSF + X2ndFlrSF))) %>% 
  group_by(YrSold) %>% summarise(n(), mean(CostPerSqft))

model_YrSold = lm(SalePrice ~ YrSold , data = df_train)
summary(model_YrSold)
# No significance
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'SaleType' as a predictor
df_train$SaleType = as.factor(df_train$SaleType)
df_train %>% select(c(SalePrice, SaleType)) %>% 
  mutate(SaleType = fct_reorder(SaleType, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=SaleType, y = SalePrice))
df_train %>% group_by(SaleType) %>% summarise(n())
model_SaleType = lm(SalePrice ~ SaleType , data = df_train)
summary(model_SaleType)
# No significance
# Do not include in the model

#  ----------------------------------------------------------------------------------
# Check for 'SaleCondition' as a predictor
df_train$SaleType = as.factor(df_train$SaleCondition)
df_train %>% select(c(SalePrice, SaleCondition)) %>% 
  mutate(SaleCondition = fct_reorder(SaleCondition, SalePrice, .fun = 'median')) %>% 
  ggplot() +geom_boxplot(aes(x=SaleCondition, y = SalePrice))
df_train %>% group_by(SaleType) %>% summarise(n())
model_SaleType = lm(SalePrice ~ SaleType , data = df_train)
summary(model_SaleType)
# No significance yet
# bin categories together
# Do Include in the model

df_train %>% filter(SalePrice>700000) %>% select(c(SalePrice, Id, SaleCondition, LotArea))

# remove insignificant variables
# df_train = df_train %>% select(-c(Street, Utilities, LotConfig, LandSlope, Condition2, OverallCond, 
                                  # RoofMatl, ExterCond, BsmtCond, BsmtUnfSF, BsmtFinSF2, Heating, 
                                  # LowQualFinSF, BsmtHalfBath, BedroomAbvGr, KitchenAbvGr, GarageYrBlt, 
                                  # GarageQual, GarageCond, X3SsnPorch, ScreenPorch, PoolArea, PoolQC, 
                                  # MiscFeature, MiscVal, MoSold, YrSold, SaleType))

# write new dataframe to csv file "output_df_train"
# write.csv(df_train, "output_df_train.csv",row.names = FALSE)

View(df_train %>% sapply(levels))
