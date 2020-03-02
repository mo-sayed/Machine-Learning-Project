# Machine Learning Project
# Boston Housing Market
# House Prices: Advanced Regression Techniques

# -------------------------------------------------------------------------------------
#                           DATA CLEANING & PREPARATION
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
#                                    TEST SET
# -------------------------------------------------------------------------------------


library('dplyr')
library('tidyr')
# library('ggplot2')
# library('lubridate')
library('forcats')
library('data.table')
library('VIM')

setwd("~/Desktop/NYC DSA/Lecture Slides/Projects/Machine Learning")

# df_train = read.csv('train.csv')
df_test = read.csv('test.csv')
# df_sample = read.csv('sample_submission.csv')

# listofvariablesd = c('Street', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'ExterCond
# BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr
# GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal
# MoSold', 'YrSold', 'SaleType')

# MSSubClass
colnames(df_test)

# Update MSZoning levels
df_test %>% group_by(MSZoning) %>% summarise(n())
# Impute the 4 missing MSZoning observations
df_test$MSZoning = kNN(df_test %>% select(c(X1stFlrSF, LotArea ,MSZoning)), k=5)$MSZoning
# df_test = df_test %>% mutate(MSZoning = ifelse(MSZoning == "C (all)", "C (all)",ifelse(MSZoning %in% c("RH", "RM"),"FV" ,ifelse(MSZoning == "RL", "RL", "FV" "RH"))))
df_test = df_test %>% mutate(MSZoning = ifelse(MSZoning == "C (all)", "C (all)", 
                                                 ifelse(MSZoning == "FV", "FV",ifelse(MSZoning == "RL", "RL", 
                                                                                      ifelse(MSZoning %in% c("RH", "RM"), "RH", "Other")))))
# LotFrontage: impute the 227 missing observations
df_test %>% filter(is.na(LotFrontage)) %>% group_by(LotFrontage) %>% summarise(n())   # 227 missing values
df_test$LotFrontage = kNN(df_test %>% select(c(LotArea, Neighborhood, LotFrontage)), k=5)$LotFrontage
# LotShape
df_test = df_test %>%  mutate(LotShape = ifelse((LotShape == "IR2" | LotShape == "IR3"), "IR2",ifelse(LotShape == "IR1", "IR1", "Reg")))
# Neighborhood
df_test = df_test %>% mutate(Neighborhood = ifelse(Neighborhood %in% c("BrDale","IDOTRR","MeadowV"), "A", ifelse(Neighborhood %in% c("BrkSide","Edwards","OldTown"), "B",
  ifelse(Neighborhood %in% c("Blueste","Mitchel","NAmes","NPkVill","SWISU","Sawyer"), "C",ifelse(Neighborhood %in% c("Blmngtn","ClearCr","CollgCr","Crawfor","Gilbert","NWAmes","SawyerW"), "D", 
    ifelse(Neighborhood %in% c("Somerst", "Timber", "Veenker" ), "E", "F"))))))
# BldgType
df_test = df_test %>%  mutate(BldgType = ifelse((BldgType %in% c("2fmCon","Duplex","Twnhs")), "C", ifelse(BldgType == "1Fam", "A", "B")))
# HouseStyle
df_test = df_test %>% mutate(HouseStyle = ifelse(HouseStyle %in% c("1.5Unf", "1.5Fin", "2.5Unf", "SFoyer"), "A",ifelse(HouseStyle %in% c("1Story", "SLvl"), "B", "C")))
# YearBuilt
# Creating new Variable 'SoldAfterBuilt' which is the difference between YrSold and YearBuilt
df_test = df_test %>% mutate(YrDiffSoldBuilt = (as.numeric(YrSold) - as.numeric(YearBuilt)))

# YearRemodAdd
df_test$YrDiffRemodBlt = df_test$YearRemodAdd - df_test$YearBuilt
df_test$YrDiffSoldRemod = df_test$YrSold - df_test$YearRemodAdd

# Exterior1st
df_test$Exterior1st = kNN(df_test %>% select(c(LotArea, Neighborhood, BldgType, HouseStyle, Exterior1st)), k=10)$Exterior1st
df_test = df_test %>% mutate(Exterior1st = ifelse(Exterior1st %in% c("BrkComm", "AshpShn", "CBlock", "PreCast", "AsbShng"), "A",
                                                    ifelse(Exterior1st %in% c("WdShing","Wd Sdng", "MetalSd", "Stone", "Stucco", "HdBoard"), "B",
                                                           ifelse(Exterior1st %in% c("BrkFace", "Plywood"), "C",ifelse(Exterior1st %in% c("VinylSd"), "D", "E")))))
# Exterior2nd
df_test$Exterior2nd = kNN(df_test %>% select(c(Exterior1st, LotArea, Neighborhood, BldgType, HouseStyle, Exterior2nd)), k=10)$Exterior2nd

df_test = df_test %>% mutate(Exterior2nd = ifelse(Exterior2nd %in% c("Brk Cmn", "AshpShn", "CBlock", "PreCast", "AsbShng"), "A",
                                                    ifelse(Exterior2nd %in% c("Wd Shng","Wd Sdng", "MetalSd", "Stone","Stucco", "HdBoard", "Plywood", "ImStucc", "BrkFace"), "B", "C")))
# MasVnrArea
df_test %>% filter(is.na(MasVnrArea)) %>% select(c(MasVnrArea, MasVnrType, Id))
df_test$MasVnrArea = kNN(df_test %>% select(c(BldgType, HouseStyle, Exterior1st, Exterior2nd, Neighborhood, MasVnrType, MasVnrArea)), k=10)$MasVnrArea
# MasVnrType
df_test$MasVnrType = kNN(df_test %>% select(c(BldgType, HouseStyle, Exterior1st, Exterior2nd, Neighborhood, MasVnrType)), k=10)$MasVnrType
# df_test = df_test %>% mutate(MasVnrType = ifelse((is.na(MasVnrType) | MasVnrType == "None"), "None",ifelse(MasVnrType == "BrkCmn", "BrkCmn", ifelse(MasVnrType == "BrkFace", "BrkFace","Stone"))))

# Foundation
df_test = df_test %>% mutate(Foundation = ifelse(Foundation == "PConc", "A", "B"))

# df_test = df_test %>% mutate(BsmtQual = (ifelse(BsmtQual == "Ex", "Ex", ifelse(BsmtQual == "Gd", "Gd",ifelse(BsmtQual == "TA", "TA", ifelse(BsmtQual == "Fa", "Fa", ifelse(BsmtQual == "Po", "Po", "NoBasement")))))))
# BsmtQual
df_test$BsmtQual = kNN(df_test %>% select(c(OverallQual, Foundation, BsmtFinSF1, TotalBsmtSF, BsmtFinSF2, BsmtQual)), k=10)$BsmtQual
# BsmtQual
df_test$BsmtCond = kNN(df_test %>% select(c(BsmtQual, OverallQual, Foundation, BsmtFinSF1, TotalBsmtSF, BsmtFinSF2, BsmtCond)), k=10)$BsmtCond

# BsmtExposure
# NAs represent no basement
df_test = df_test %>% mutate(BsmtExposure = ifelse(is.na(BsmtExposure), "NoBasement",
                                                     ifelse(BsmtExposure == "No", "No",
                                                            ifelse(BsmtExposure == "Mn", "Mn",
                                                                   ifelse(BsmtExposure == "Av", "Av", "Gd")))))
# BsmtFinType1
# BsmtFinType1   &   BsmtFinType2   NAs --> "No Basement"
df_test = df_test %>% mutate(BsmtFinType1 = ifelse(is.na(BsmtFinType1), "No Basement", 
                                                     ifelse(BsmtFinType1 == "GLQ", "A", "B")))
df_test = df_test %>% mutate(BsmtFinType2 = ifelse(is.na(BsmtFinType2), "No Basement", 
                                                     ifelse(BsmtFinType2 == "GLQ", "A", "B")))
# BsmtFinSF1   & BsmtFinSF2
df_test$BsmtFinSF1 = kNN(df_test %>% select(c( BldgType, TotalBsmtSF, BsmtQual, BsmtExposure, BsmtFinType1, BsmtFinSF1)), k=10)$BsmtFinSF1
df_test$BsmtFinSF2 = kNN(df_test %>% select(c(TotalBsmtSF, BsmtQual, BsmtExposure, BsmtFinType1, BsmtFinSF1, BsmtFinSF2)), k=10)$BsmtFinSF2

# df_test %>% filter(is.na(BsmtFinSF1)) %>% select(c(Id, BldgType, LandSlope, LotConfig, LandContour ,Foundation, BsmtFinSF1, TotalBsmtSF, BsmtQual, BsmtExposure, BsmtFinType1))
# df_test %>% filter(Id == 2121) %>% select(c(Id, BldgType, LandSlope, LotConfig, LandContour ,Foundation, BsmtFinSF1, TotalBsmtSF, BsmtQual, BsmtExposure, BsmtFinType1))

# TotalBsmtSF
df_test %>% filter(is.na(TotalBsmtSF)) %>% select(c(Id, BsmtFinType1))
df_test$TotalBsmtSF = kNN(df_test %>% select(c( BldgType, TotalBsmtSF, BsmtQual, BsmtExposure, BsmtFinType1, BsmtFinSF1)), k=10)$TotalBsmtSF


# HeatingQC
df_test %>% filter(is.na(Electrical)) %>% summarise(n())
df_test = df_test %>% mutate(HeatingQC = (ifelse(((HeatingQC == "Fa")|(HeatingQC == "Po")), "Bad",ifelse(((HeatingQC == "TA")|(HeatingQC == "Gd")), "Good", "Excellent"))))
# CentralAir
# Electrical
df_test$Electrical = kNN(df_test %>% select(c(YearBuilt, YearRemodAdd, Electrical)), k=10)$Electrical
df_test = df_test %>% mutate(Electrical = ifelse(Electrical == "SBrkr", "A", "B"))
# X1stFlrSF
# X2ndFlrSF
# X2ndFlrYN
for (i in (1:nrow(df_test))){
  df_test$X2ndFlrYN[i] = (ifelse(df_test$X2ndFlrSF[i] == 0, "No", "Yes"))
}
# FullBath
# Create new variable of Total Baths
df_test %>% filter(is.na(BsmtFullBath) | is.na(BsmtHalfBath)) %>% select(c(Id, BsmtFullBath,BsmtFinType1, FullBath))
# df_test$Electrical = kNN(df_test %>% select(c(YearBuilt, YearRemodAdd, Electrical)), k=10)$Electrical
df_test$BsmtFullBath = kNN(df_test %>% select(c(BsmtFinSF1, TotalBsmtSF, BsmtExposure, FullBath, TotRmsAbvGrd, BsmtFullBath )), k=10)$BsmtFullBath
df_test$BsmtHalfBath = kNN(df_test %>% select(c(BsmtFinSF1, TotalBsmtSF, BsmtExposure, FullBath, HalfBath, TotRmsAbvGrd, BsmtHalfBath )), k=10)$BsmtHalfBath
df_test$TotalBath = (df_test$FullBath + df_test$BsmtFullBath + 0.5*df_test$BsmtHalfBath + 0.5*df_test$HalfBath)
# HalfBath
for (i in (1:nrow(df_test))){
  df_test$HalfBathYN[i] = (ifelse(df_test$HalfBath[i] == 0, "No", "Yes"))
}
df_test$HalfBathYN = as.factor(df_test$HalfBathYN)
# KitchenQual
df_test %>% filter(is.na(KitchenQual)) %>% select(c(Id, KitchenQual, KitchenAbvGr))
df_test$KitchenQual = kNN(df_test %>% select(c(KitchenAbvGr, OverallQual, OverallCond, TotRmsAbvGrd, KitchenQual )), k=10)$KitchenQual

# TotRmsAbvGrd
# Functional
# df_test = df_test %>% mutate(Functional = ifelse((Functional == "Typ"), "Typ", "Other"))
# Fireplaces
for (i in (1:nrow(df_test))){
  df_test$Fireplaces[i] = (ifelse(df_test$Fireplaces[i] == 0, "No", "Yes"))
}

# FireplaceQu
# GarageType
df_test = df_test %>% mutate(GarageType = ifelse(is.na(GarageType), "No Garage",
                                                   ifelse(GarageType == "2Types", "2Types",ifelse(GarageType == "Attchd", "Attchd",
                                                                                                  ifelse(GarageType == "Basment", "Basment", ifelse(GarageType == "BuiltIn", "BuiltIn", ifelse(GarageType == "CarPort", "CarPort", "Detchd")))))))

# GarageYrBlt
df_test$GarageYrBlt =kNN(df_test %>% select(c(YearBuilt, YearRemodAdd, GarageType, GarageCond, GarageQual, GarageYrBlt)), k=10 )$GarageYrBlt
df_test$YrDiffGarageBlt = df_test$GarageYrBlt - df_test$YearBuilt

# GarageFinish
df_test = df_test %>% mutate(GarageFinish = ifelse(is.na(GarageFinish), "No Garage",
                                                     ifelse(GarageFinish == "Fin", "Fin",
                                                            ifelse(GarageFinish == "RFn", "RFn", "Unf"))))
# GarageCars
df_test %>% filter(is.na(GarageCars)) %>% select(c(Id, GarageCars, GarageType))
df_test %>% filter(Id == 2577) %>% select(c(Id, GarageCars, GarageType))
df_test$GarageCars = kNN(df_test %>% select(c(GarageYrBlt, GarageType, YearBuilt, PavedDrive,TotRmsAbvGrd , GarageCars )), k=10)$GarageCars

# GarageArea
df_test %>% filter(is.na(GarageArea)) %>% select(c(Id, GarageArea, GarageType))
df_test$GarageArea = kNN(df_test %>% select(c(GarageCars, GarageYrBlt, GarageType, YearBuilt, LotArea,LotFrontage ,PavedDrive,TotRmsAbvGrd , GarageArea )), k=10)$GarageArea

# GarageQual
df_test = df_test %>% mutate(GarageQual = ifelse(is.na(GarageQual), "No Garage",
                                                   ifelse(GarageQual == "Ex", "Ex",ifelse(GarageQual == "Fa", "Fa",
                                                                                          ifelse(GarageQual == "Gd", "Gd", ifelse(GarageQual == "Po", "Po", "TA"))))))

# GarageCond
df_test = df_test %>% mutate(GarageCond = ifelse(is.na(GarageCond), "No Garage",
                                                   ifelse(GarageCond == "Ex", "Ex",ifelse(GarageCond == "Fa", "Fa",
                                                                                          ifelse(GarageCond == "Gd", "Gd", ifelse(GarageCond == "Po", "Po", "TA"))))))
# PavedDrive
df_test %>% group_by(PavedDrive) %>% summarise(n())
df_test = df_test %>% mutate(PavedDrive = ifelse(PavedDrive == "Y", "Paved", "PartialNo"))
# WoodDeckSF
df_test = df_test %>% mutate(FrontAreaSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + X3SsnPorch + ScreenPorch)
# OpenPorchSF
# EnclosedPorch
# X3SsnPorch

# ScreenPorch

# PoolArea

# PoolQC

# Fence
df_test %>% group_by(Fence) %>% summarise(n())
df_test = df_test %>% mutate(Fence = ifelse(is.na(Fence), "No Fence",
                                              ifelse(Fence == "GdPrv", "GdPrv",ifelse(Fence == "GdWo", "GdWo",
                                                                                      ifelse(Fence == "MnPrv", "MnPrv", "MnWw")))))


# MiscFeature

# MiscVal

# MoSold

# YrSold

# SaleType
df_test = df_test %>% mutate(SaleType = ifelse(SaleType %in% c("Oth", "ConLI", "COD", "ConLD", "ConLW"), "A",
                                      ifelse(SaleType %in% c("WD", "CWD", "VWD"), "B", "C")))

# SaleCondition

# SalePrice

# listofvariablestobed = c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'Exterior2nd', 'ExterCond
# BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr
# GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal
# MoSold', 'YrSold', 'SaleType', 'SaleCondition')

# c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'OverallCond', 'RoofMatl', 'Exterior2nd', 'ExterCond', 'BsmtCond', 'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageYrBlt', 'GarageQual', 'GarageCond', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition')

df_test = df_test %>% select(-c('Street', 'Alley', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'YearBuilt',
                                'Condition2', 'OverallCond', 'RoofStyle','RoofMatl', 'ExterCond', 'YearRemodAdd',
                                'BsmtCond','BsmtFinType2' ,'BsmtUnfSF', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 
                                'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr','Functional','FireplaceQu', 'GarageYrBlt',
                                'GarageQual', 'GarageCond','WoodDeckSF','OpenPorchSF','EnclosedPorch' ,'X3SsnPorch', 
                                'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'CentralAir', 'YrSold', 
                                'MiscFeature', 'MiscVal', 'MoSold', 'SaleCondition'))

# write new dataframe to csv file "output_df_train"
write.csv(df_test, "output_df_test.csv",row.names = FALSE)

# use to check for any missing values overall in dataframe
# View(df_test %>% sapply(function(x) sum(is.na(x))))
