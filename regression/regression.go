package regression

import (
	"math"
	"proj3/data"
)

// Calculates MSE, our loss function
func CalcMSE(predicted []float64, actual []float64) float64{
	mse := float64(0)
	for i := 0; i < len(predicted); i++ {
		mse += math.Pow(predicted[i] - actual[i], 2)
	}
	return mse / float64(len(predicted))
}

// Forecasts linear regrssion given parameters
func Forecast(mu float64, beta float64, x []float64) []float64 {
	predicted := make([]float64, 0)
	for i := 0; i < len(x); i++ {
		predicted = append(predicted, beta * x[i] + mu)
	}
	return predicted
}

// Calculates the gradient of the cost function with respect to beta, which is -(2/n)*sum(X(Y-Yhat))
func calcGradientBeta(predicted []float64, data data.InputData) float64{
	gradientBeta := float64(0)
	for i := 0; i < len(predicted); i++{
		gradientBeta += ((data.Y[i] - predicted[i]) * data.X[i])
	}
	gradientBeta = -(float64(2) * gradientBeta / float64(len(predicted)))
	return gradientBeta
}

// Calculates the gradient of the cost function with respect to mu, which is -(2/n)*sum(Y-Yhat)
func calcGradientMu(predicted []float64, actual []float64) float64{
	gradientMu := float64(0)
	for i := 0; i < len(predicted); i++{
		gradientMu += (actual[i] - predicted[i])
	}
	gradientMu = -(float64(2)* gradientMu / float64(len(predicted)))
	return gradientMu
}

// Updates parameters per descent. Important that both parameters are updated simultaneously (ie do not update predicted until all parameters are updated)
func UpdateParams(parameters Parameters, data data.InputData, alpha float64) Parameters {
	predicted := Forecast(parameters.Mu, parameters.Beta, data.X)
	parameters.Mu -= alpha * calcGradientMu(predicted, data.Y)
	parameters.Beta -= alpha * calcGradientBeta(predicted, data)
	return parameters
}

// Normalizes independent data. Need to feature scale in order for our algorithm to be able to handle gradient descent at different magnitudes without having to scale alpha
func Normalize(rawData data.InputData, minX float64, maxX float64) data.InputData {
	var dataNormalized data.InputData
	dataNormalized.X = make([]float64, 0)
	dataNormalized.Y = rawData.Y
	for i:=0; i < len(rawData.X); i++{
		dataNormalized.X = append(dataNormalized.X, (rawData.X[i] - minX)/ (maxX - minX))
	}
	return dataNormalized
}

// Denormalizes our parameters, which are calibrated on normalized data
func UnNormalize (parameters Parameters, data data.InputData, minX float64, maxX float64) Parameters {
	//in order to grab our correct beta on unnormalized data, we need to unnormalize beta
	parameters.Beta = parameters.Beta /(maxX - minX) - minX
	return parameters
}

// Calculates the min and max of a slice
// This function is from StackExchange: https://stackoverflow.com/questions/34259800/is-there-a-built-in-min-function-for-a-slice-of-int-arguments-or-a-variable-numb
func MinMax (arr []float64) (float64, float64) {
	var max float64 = arr[0]
	var min float64 = arr[0]
	for _, value := range arr {
		if max < value {
			max = value
		}
		if min > value {
			min = value
		}
	}
	return min, max
}

// Member variables represent the intercept and coefficient of our univariate regression model
type Parameters struct {
	Mu float64
	Beta float64
}
