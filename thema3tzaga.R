myData <- data.frame(y=numeric(0), x1=numeric(0),
                     x2=numeric(0),
                     x3=numeric(0),
                     x4=numeric(0),
                     x5=numeric(0),
                     x6=numeric(0))
for (i in 1:4){
  myData[i,] <- runif(7, min=1, max=10)
}
rModel<-lm( y ~ ., data=myData)
print(rModel$coefficients)

class(rModel)

model_1<-lm( y ~ ., data=myData)
vars=myData
cor(vars)
