#6.7.1 PCR
#load data
library(ISLR)
fix(Hitters)
names(Hitters)


#remove nans
#I think there is another method complete.cases that returns only thos cases with all varaibles diff from nan
sum(is.na(Hitters$Salary))
Hitters = na.omit(Hitters)
sum(is.na(Hitters$Salary))

library(pls)
set.seed(2)
pcr.fit = pcr(Salary~., data = Hitters, scale = TRUE, validation = "CV")

summary(pcr.fit)

validationplot(pcr.fit,val.type = "MSEP")



######obtain plsr
set.seed(1)
train = sample(c(TRUE,FALSE),nrow(Hitters),rep = TRUE)
test = (!train)
pls.fit = plsr(Salary~. , data = Hitters, subset = train,scale = TRUE,validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type= "MSEP")
