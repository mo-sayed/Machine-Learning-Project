# Machine Learning Project
# Boston Housing Market
# House Prices: Advanced Regression Techniques

# -------------------------------------------------------------------------------------
#                           DATA CLEANING & PREPARATION
# -------------------------------------------------------------------------------------

library('dplyr')
library('tidyr')
library('ggplot2')
library('lubridate')
library('forcats')
library('data.table')
library('VIM')

# setwd("~/Desktop/NYC DSA/Lecture Slides/Projects/Machine Learning")

df_train = read.csv('train.csv')
df_test = read.csv('test.csv')
df_sample = read.csv('sample_submission.csv')

# listofvariablesremoved = c('Street', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'ExterCond
# BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr
# GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal
# MoSold', 'YrSold', 'SaleType')

# Id
# MSSubClass
# MSZoning
df_train = df_train %>% mutate(MSZoning = ifelse(MSZoning == "C (all)", "C (all)",
          ifelse(MSZoning == "FV", "FV",ifelse(MSZoning == "RL", "RL", "RH"))))
# LotFrontage
# LotFrontage_Outliers -> c(1299', '935)
# df_train = df_train %>% mutate(is.na(LotFrontage), "0", "LotFrontage")
df_train = df_train %>% mutate(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage))
df_train = df_train %>% filter(LotFrontage < 300 | is.na(LotFrontage))
# LotArea
# LotArea -> c(314', '314', '692', '1183)
df_train = df_train %>% filter(LotArea < 200000)
df_train = df_train %>% filter(SalePrice < 700000)
# Street
REMOVE
# Alley
REMOVE
# LotShape
df_train = df_train %>%  mutate(LotShape = ifelse((LotShape == "IR2" | LotShape == "IR3"), "IR2",
            ifelse(LotShape == "IR1", "IR1", "Reg")))
# LandContour
# Utilities
REMOVE
# LotConfig
REMOVE
# LandSlope
REMOVE
# Neighborhood

# Condition1
REMOVE
# Condition2
REMOVE
# BldgType
df_train = df_train %>%  mutate(BldgType = ifelse((BldgType == "2fmCon" | BldgType == "Duplex" | BldgType == "Twnhs"), "Other", 
                         ifelse(BldgType == "1Fam", "1Fam", "TwnhsE")))

# HouseStyle
df_train = df_train %>% mutate(HouseStyle = ifelse(HouseStyle == "1Story", "1Story",
             ifelse((HouseStyle == "2Story" | HouseStyle == "2.5Fin"), "2+Fin", 
             ifelse(HouseStyle == "SLvl", "SLvl", "Other"))))
# OverallQual
# OverallCond
REMOVE
# YearBuilt
# YearRemodAdd
# RoofStyle
# RoofMatl
REMOVE
# Exterior1st
# Exterior2nd
# MasVnrType
df_train = df_train %>% mutate(MasVnrType = ifelse((is.na(MasVnrType) | MasVnrType == "None"), "None",
                   ifelse(MasVnrType == "BrkCmn", "BrkCmn", 
                    ifelse(MasVnrType == "BrkFace", "BrkFace","Stone"))))
# MasVnrArea
df_train = df_train %>% mutate(MasVnrArea = ifelse(((is.na(MasVnrArea)|
            MasVnrArea <=1)), 0.00001, MasVnrArea))
# ExterQual
# ExterCond
REMOVE
# Foundation
df_train = df_train %>% mutate(BsmtQual = (ifelse(BsmtQual == "Ex", "Ex", ifelse(BsmtQual == "Gd", "Gd",
      ifelse(BsmtQual == "TA", "TA", ifelse(BsmtQual == "Fa", "Fa", ifelse(BsmtQual == "Po", "Po", "NoBasement")))))))
# BsmtQual
# BsmtCond
REMOVE
# BsmtExposure
df_train = df_train %>% mutate(BsmtExposure = (ifelse(((BsmtExposure == "No")|(is.na(BsmtExposure))), "No Basement",
         ifelse(BsmtExposure == "Mn", "Mn", ifelse(BsmtExposure == "Av", "Av", "Gd")))))
# BsmtFinType1
df_train = df_train %>% mutate(BsmtFinType1 = (ifelse(((BsmtFinType1 == "Rec")|(BsmtFinType1 == "BLQ")|(BsmtFinType1 == "LwQ")|(BsmtFinType1 == "ALQ")), "AvQ",
                                                      ifelse(BsmtFinType1 == "GLQ", "GLQ",
                                                             ifelse(BsmtFinType1 == "Unf", "Unf", "NoBasement")))))
df_train = df_train %>% mutate(BsmtFinType1 = (ifelse(is.na(BsmtFinType1), "NoBasement", BsmtFinType1)))
# BsmtFinType2
df_train = df_train %>% mutate(BsmtFinType2 = (ifelse(((BsmtFinType2 == "Rec")|(BsmtFinType2 == "BLQ")|(BsmtFinType2 == "LwQ")), "BLQ",
    ifelse(((BsmtFinType2 == "GLQ")|(BsmtFinType2 == "ALQ")), "GLQ", ifelse(BsmtFinType2 == "Unf", "Unf", "NoBasement")))))
df_train = df_train %>% mutate(BsmtFinType2 = (ifelse(is.na(BsmtFinType2), "NoBasement", BsmtFinType2)))
# BsmtFinSF1
# BsmtFinSF2    REMOVE
df_train$BsmtFinSF2 = ifelse(((df_train$BsmtFinSF2 == 0) & (df_train$BsmtFinSF1 > 0)), df_train$BsmtUnfSF, df_train$BsmtFinSF2)
ratio_SF1 = df_train %>% summarise(sum(BsmtFinSF1)/ (sum(BsmtFinSF1)+sum(BsmtFinSF2)))
ratio_SF2 = df_train %>% summarise(sum(BsmtFinSF2)/ (sum(BsmtFinSF1)+sum(BsmtFinSF2)))
for (i in (1:nrow(df_train))){
  df_train$BsmtFinSF1[i] = ifelse( ((df_train$BsmtFinSF1[i] == 0)&(df_train$BsmtFinSF2[i] == 0)&
  (df_train$BsmtFinType1[i] == "Unf")), round(ratio_SF1*df_train$BsmtUnfSF[i]), df_train$BsmtFinSF1[i])
  df_train$BsmtFinSF2[i] = ifelse( ((df_train$BsmtFinSF2[i] == 0)&(df_train$BsmtFinType1[i] == "Unf")), 
    round(ratio_SF2*df_train$TotalBsmtSF[i]), df_train$BsmtFinSF2[i])
}
df_train$BsmtFinSF1 = as.numeric(unlist(df_train$BsmtFinSF1))
df_train$BsmtFinSF2 = as.numeric(unlist(df_train$BsmtFinSF2))
# BsmtUnfSF
REMOVE
# TotalBsmtSF
# Heating
REMOVE
# HeatingQC
df_train = df_train %>% mutate(HeatingQC = (ifelse(((HeatingQC == "Fa")|(HeatingQC == "Po")), "Bad",
          ifelse(((HeatingQC == "TA")|(HeatingQC == "Gd")), "Good", "Excellent"))))
# CentralAir
# Electrical
df_train$Electrical = kNN(df_train %>% select(c(SalePrice, Electrical)), k=5)$Electrical
df_train = df_train %>% mutate(Electrical = ifelse((Electrical == "FuseF"| Electrical == "FuseP" | 
  Electrical == "Mix"), "FuseB", ifelse(Electrical == "FuseA","FuseG", "SBrkr")))
# X1stFlrSF
df_train = df_train %>% filter(X1stFlrSF <4000)
# X2ndFlrSF
# LowQualFinSF
REMOVE
# GrLivArea
# BsmtFullBath
# BsmtFullBath_Outliers -> c(739)
df_train = df_train %>% filter(BsmtFullBath <= 2)
REMOVE
# BsmtHalfBath
REMOVE
# FullBath
# Create new variable of full bath
df_train$TotalBath = (df_train$FullBath + df_train$BsmtFullBath + 0.5*df_train$BsmtHalfBath + 0.5*df_train$HalfBath)
# HalfBath
REMOVE
# BedroomAbvGr
REMOVE
# KitchenAbvGr
REMOVE
# KitchenQual
# TotRmsAbvGrd
# TotRmsAbvGrd_Outliers = 636
df_train = df_train %>% filter(TotRmsAbvGrd <= 13)
# Functional
df_train = df_train %>% mutate(Functional = ifelse((Functional == "Typ"), "Typ", "Other"))
# Fireplaces
# FireplaceQu
df_train = df_train %>% mutate(FireplaceQu = ifelse(is.na(FireplaceQu), "None",ifelse(FireplaceQu == "Ex", "Ex",
ifelse(FireplaceQu == "Fa", "Fa",ifelse(FireplaceQu == "Gd", "Gd", ifelse(FireplaceQu == "Po", "Po", "TA"))))))
# GarageType
df_train = df_train %>% mutate(GarageType = ifelse(is.na(GarageType), "No Garage", 
ifelse(GarageType == "2Types", "2Types",ifelse(GarageType == "Attchd", "Attchd",
ifelse(GarageType == "Basment", "Basment", ifelse(GarageType == "BuiltIn", "BuiltIn",
ifelse(GarageType == "CarPort", "CarPort", "Detchd")))))))
# GarageYrBlt
REMOVE
# GarageFinish
df_train = df_train %>% mutate(GarageFinish = ifelse(is.na(GarageFinish), "No Garage",
ifelse(GarageFinish == "Fin", "Fin",ifelse(GarageFinish == "RFn", "RFn", "Unf"))))
# GarageCars
# GarageArea
# GarageQual
REMOVE
# GarageCond
REMOVE
# PavedDrive
df_train = df_train %>% mutate(PavedDrive = ifelse(PavedDrive == "Y", "Paved", "PartialNo"))
# WoodDeckSF
df_train = df_train %>% mutate(WoodDeckSF = ifelse(WoodDeckSF == 0, 0.01, WoodDeckSF))
# OpenPorchSF
df_train = df_train %>% mutate(OpenPorchSF = ifelse(OpenPorchSF == 0, 0.01, OpenPorchSF))
# EnclosedPorch
df_train = df_train %>% mutate(EnclosedPorch = ifelse(EnclosedPorch == 0, 0.01, EnclosedPorch))
# X3SsnPorch
REMOVE
# ScreenPorch
REMOVE
# PoolArea
REMOVE
# PoolQC
REMOVE
# Fence
REMOVE
# MiscFeature
REMOVE
# MiscVal
REMOVE
# MoSold
REMOVE
# YrSold
REMOVE
# SaleType
REMOVE
# SaleCondition
REMOVE
# SalePrice

# listofvariablestoberemoved = c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'Exterior2nd', 'ExterCond
# BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr
# GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal
# MoSold', 'YrSold', 'SaleType', 'SaleCondition')

# c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'Exterior2nd', 'ExterCond', 'BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition')

df_train = df_train %>% select(-c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'Exterior2nd', 'ExterCond', 'BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'))

# write new dataframe to csv file "output_df_train"
write.csv(df_train, "output_df_train.csv",row.names = FALSE)

