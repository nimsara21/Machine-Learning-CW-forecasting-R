library(fpp2)
library(readxl)
library(Metrics)
library(MLmetrics)
library(neuralnet)
library(tsDyn)
library(vars)
library(quantmod)
library(forecast)
library(nets)


rm(list=ls())

#load data
data <- read.csv("D:/IIT/Year 2/Sem 2/Machine Learning/CW/UoW_load (1).csv")
View(data)

#dividing the data set into training and testing data
dataTesting = tail(data, n =70)
dataTraining = head(data, n =430)



normalization <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


#Scaling data.
dataSc <- as.data.frame(lapply(data[2:4], normalization))
dataScaled <- dataSc
dataSc <- cbind(dataSc, data[c(4)])
View(dataSc)

#naming the attributes.
names(dataSc)[1] <- "Date"
names(dataSc)[2] <- "Nine"
names(dataSc)[3] <- "Ten"
names(dataSc)[4] <- "Eleven"

dataTestingSc = tail(dataSc, n =70)
dataTrainingSc = head(dataSc, n =430)

#Creating th function to reverse the normalization function
renormalizing <- function(x, min, max) { 
  return( (max - min)*x + min )
}


#getting the min and max values of the 4th column of the dataset.
minV <- min(dataSc[,4])
maxV <- max(dataSc[,4])


#Creating Neural Network models with AR

#First NN Model with 3 , 4 hidden layers 
NN1<- neuralnet(Eleven ~ Eleven, hidden = c(3, 4),
                data = dataTrainingSc ,linear.output=TRUE)


plot(NN1)
NN1$result.matrix
NN1$weights
#Plotting First NN Model

#Model Performance
pfModel <- predict(NN1, dataTestingSc[3])
pfModel

#re normalizing the predicted values.
renomalizingVal1 <- renormalizing(pfModel, minV, maxV)
renomalizingVal1 = unlist(as.list(renomalizingVal1),recursive=F)
renomalizingVal1
plot(pfModel)


plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal1, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)




#RMSE
RMSE(renomalizingVal1,dataTesting[,4])
#MSE
MSE(renomalizingVal1,dataTesting[,4])
#MAPE
mape(renomalizingVal1,dataTesting[,4])



#Second NN Model with 3 , 10 hidden layers 
NN2 <- neuralnet(Eleven ~ Eleven, hidden = c(3, 10),
                 data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN2
plot(NN2)



#Model Performance
pfModel2 <- predict(NN2, dataTestingSc[3])
pfModel2


#re normalizing the predicted values.
renomalizingVal2 <- renormalizing(pfModel2, minV, maxV)
renomalizingVal2 = unlist(as.list(renomalizingVal2),recursive=F)
renomalizingVal2

plot(pfModel2)


plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal1, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal2,dataTesting[,4])
#MSE
MSE(renomalizingVal2,dataTesting[,4])
#MAPE
mape(renomalizingVal2,dataTesting[,4])



#Third NN Model with 3 , 10 hidden layers and t -1 input 
NN3 <- neuralnet(Eleven ~ Eleven - 1, hidden = c(3, 10),
                 data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN3
plot(NN3)

#Model Performance
pfModel3 <- predict(NN3, dataTestingSc[3])
pfModel3

#re normalizing the predicted values.
renomalizingVal3 <- renormalizing(pfModel3, minV, maxV)
renomalizingVal3 = unlist(as.list(renomalizingVal3),recursive=F)
renomalizingVal3

plot(pfModel3)


plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal3, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal3,dataTesting[,4])
#MSE
MSE(renomalizingVal3,dataTesting[,4])
#MAPE
mape(renomalizingVal3,dataTesting[,4])





#Fourth NN Model with 3 , 4 hidden layers and t -1 input 
NN4 <- neuralnet(Eleven ~ Eleven - 1, hidden = c(3, 4),
                 data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN4
plot(NN4)

#Model Performance
pfModel4 <- predict(NN4, dataTestingSc[3])
pfModel4

#re normalizing the predicted values.
renomalizingVal4 <- renormalizing(pfModel4, minV, maxV)
renomalizingVal4 = unlist(as.list(renomalizingVal4),recursive=F)
renomalizingVal4


plot(pfModel4)


plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal4, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal4,dataTesting[,4])
#MSE
MSE(renomalizingVal4,dataTesting[,4])
#MAPE
mape(renomalizingVal4,dataTesting[,4])





#Fifth NN Model with 3 , 8 hidden layers and t - 2 input 
NN5 <- neuralnet(Eleven ~ Eleven + Date, hidden = c(3, 8),
                 data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN5
plot(NN5)

#Model Performance
pfModel5 <- predict(NN5, dataTestingSc[1:2])
pfModel5

#re normalizing the predicted values.
renomalizingVal5 <- renormalizing(pfModel5, minV, maxV)
renomalizingVal5 = unlist(as.list(renomalizingVal5),recursive=F)
renomalizingVal5
plot(pfModel5)



plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal5, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal5,dataTesting[,4])
#MSE
MSE(renomalizingVal5,dataTesting[,4])
#MAPE
mape(renomalizingVal5,dataTesting[,4])




#Fifth NN Model with 5 hidden layers and t - 1 input 
NN6 <- neuralnet(Eleven ~ Eleven - 1 + Date, hidden = c(5),
                 data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN6
plot(NN6)
#Model Performance
pfModel6 <- predict(NN6, dataTestingSc[1:2])
pfModel6
#re normalizing the predicted values.
renomalizingVal6 <- renormalizing(pfModel6, minV, maxV)
renomalizingVal6 = unlist(as.list(renomalizingVal6),recursive=F)
renomalizingVal6

plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal6, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal6,dataTesting[,4])
#MSE
MSE(renomalizingVal6,dataTesting[,4])
#MAPE
mape(renomalizingVal6,dataTesting[,4])



########################################

#Creating Neural Network models with NARX

#First NN Model(NARX) with 3, 5 hidden layers 
NN7<- neuralnet(Eleven ~ Date + Nine + Ten + Eleven, hidden=c(3, 5) ,
                data = dataTrainingSc ,linear.output=TRUE)

#Plotting NN7
plot(NN7)

#Model Performance
pfModel7 <- predict(NN7, dataTestingSc[1:4])
pfModel7
#re normalizing the predicted values.
renomalizingVal7 <- renormalizing(pfModel7, minV, maxV)
renomalizingVal7 = unlist(as.list(renomalizingVal7),recursive=F)
renomalizingVal7


plot(pfModel7)
plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal7, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal7,dataTesting[,4])
#MSE
MSE(renomalizingVal7,dataTesting[,4])
#MAPE
mape(renomalizingVal7,dataTesting[,4])





#Second NN Model(NARX) with 5, 10 hidden layers 
NN8<- neuralnet(Eleven ~ Date + Nine + Ten + Eleven, hidden=c(5, 10) ,
                data = dataTrainingSc ,linear.output=TRUE)

#plotting NN8
plot(NN8)

#Model Performance
pfModel8 <- predict(NN8, dataTestingSc[1:4])
plot(pfModel8)
#re normalizing the predicted values.
renomalizingVal8 <- renormalizing(pfModel8, minV, maxV)
renomalizingVal8 = unlist(as.list(renomalizingVal8),recursive=F)
renomalizingVal8


plot(pfModel8)
plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal8, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)



#RMSE
RMSE(renomalizingVal8,dataTesting[,4])
#MSE
MSE(renomalizingVal8,dataTesting[,4])
#MAPE
mape(renomalizingVal8,dataTesting[,4])




#Second NN Model(NARX) with 3, 8 hidden layers 
NN9<- neuralnet(Eleven ~ Date + Nine + Ten + Eleven, hidden=c(3, 8) ,
                data = dataTrainingSc ,linear.output=TRUE)

#plotting NN9
plot(NN9)

#Model Performance
pfModel9 <- predict(NN9, dataTestingSc[1:4])
pfModel9
#re normalizing the predicted values.
renomalizingVal9 <- renormalizing(pfModel9, minV, maxV)
renomalizingVal9 = unlist(as.list(renomalizingVal9),recursive=F)
renomalizingVal9



#Second NN Model(NARX) with 3, 8 hidden layers with t -1 input
NN10<- neuralnet(Eleven ~ Date + Nine + Ten + Eleven - 1, hidden=c(3, 8) ,
                 data = dataTrainingSc ,linear.output=TRUE)

#plotting NN10
plot(NN10)

#Model Performance
pfModel10 <- predict(NN10, dataTestingSc[1:4])
pfModel10
#re normalizing the predicted values.
renomalizingVal10 <- renormalizing(pfModel10, minV, maxV)
renomalizingVal10 = unlist(as.list(renomalizingVal10),recursive=F)
renomalizingVal10


plot.ts(renomalizingVal10)

#RMSE
RMSE(renomalizingVal10,dataTesting[,4])
#MSE
MSE(renomalizingVal10,dataTesting[,4])
#MAPE
mape(renomalizingVal10,dataTesting[,4])





plot(dataTestingSc[,4] , ylab = "Predicted vs Expected", type="l", col="red" )
par(new=TRUE)
plot(renomalizingVal1, ylab = " ", yaxt="n", type="l", col="green" ,main='Predicted 
Value Vs Expected Value Graph')
legend("topleft",
       c("Expected","Predicted"),
       fill=c("red","green")
)

plot(x = NN8, y = NN8$weights,
     xlab="PV", ylab="actV")



#RMSE
RMSE(renomalizingVal5,dataTesting[,4])
#MSE
MSE(renomalizingVal5,dataTesting[,4])
#MAPE
mape(renomalizingVal5,dataTesting[,4])

