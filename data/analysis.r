
close1.data <- read.csv(file="close_1_returns.csv", header=TRUE, sep=",")
close5.data <- read.csv(file="close_5_returns.csv", header=TRUE, sep=",")
close20.data <- read.csv(file="close_20_returns.csv", header=TRUE, sep=",")

library(psych)
describe(close1.data)
describe(close5.data)
describe(close20.data)
