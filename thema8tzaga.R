x = rnorm(20) 
y = 3 + 3*x + 190*x^2 + 250*x^3 +200*x^4 +  rnorm(length(x),0,50) 
z = x^2
f = x^3
p = x^4

linearmodel = lm(y~x + z + f + p) #model fitting
summary(linearmodel)


calculateRMSE<-function(predictedValues, actualValues){
  err<- sqrt( mean((actualValues - predictedValues)^2)  )
  return( err )
}

kFoldCrossValidation<-function(data, frml, k){
  dataset<-data[sample(nrow(data)),]
  folds <- cut(seq(1,nrow(dataset)), breaks=k, labels=FALSE)
  RMSE<-vector()
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    training.linear.model<-lm( frml, data = trainData)
    test.linear.model<-lm( frml, data = trainData)
    k = lm( frml, data = trainData)
    r2 = summary(k)$r.squared
    predicted<-predict(training.linear.model, trainData)
    trainingdataerror<-calculateRMSE(predicted, trainData[, "y"])
    trainerror<-c(RMSE,trainingdataerror)
    predicted<-predict(test.linear.model, testData)
    error<-calculateRMSE(predicted, testData[, "y"])
    RMSE<-c(RMSE, error)
  }
  errorlist = c(mean(RMSE),r2 , trainerror)
  return(errorlist )# Επιστροφή μέσης τιμής των σφαλμάτων που προέκυψαν απ'όλα τα τμήματα ελέγχου
}

modelMeanRMSE<-vector()
lamodel = cbind.data.frame(y,x,z,f,p)
predictionModels<-vector()
predictionModels[1]<-"y ~ x + z  + f + p "

for (k in 1:length(predictionModels)){
  modelErr<-kFoldCrossValidation(lamodel, as.formula(predictionModels[k]), 10)
  modelMeanRMSE<-c(modelMeanRMSE, modelErr)
  print( sprintf("Linear regression model [%s]: generalization error [%f]", predictionModels[k], modelErr[1] ) )
  print( sprintf("R spuared for the training data is [%f]", modelErr[2] ) )
  print( sprintf("training error is [%f]", modelErr[3] ) )
}
plot(x,y, xlab = "X (random variable)", ylab = "Y", main = "Graphic representation of Overfitting in our random data" ) #plot for data
smoothspline = smooth.spline(x,y,df = 20) # line fitting
lines(smoothspline, col = "green")
