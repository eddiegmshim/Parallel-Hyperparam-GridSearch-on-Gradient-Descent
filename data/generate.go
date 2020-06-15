package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
)

// Generates data of sample size n, where the dependent variable is simply the independent variable * 5 + 100 + noise
func GenerateTrainingData(n int, outputFilePath string){
	trueBeta := float64(5)
	trueMu := float64(100)
	trueErrorVariance := float64(25)
	fmt.Println("Generating data into", outputFilePath)
	file, err := os.Create(outputFilePath)
	if err!= nil {
		log.Fatal("Error: could not create file")
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	for i:=0; i < n; i++ {
		x := rand.Float64() * float64(100)
		noise := rand.NormFloat64() * trueErrorVariance + 0 // randomly drawing error from ~N(0, trueErrorVariance)
		y := trueBeta * x + trueMu + noise
		row := []string{ fmt.Sprintf("%f", x),fmt.Sprintf("%f", y)}
		err := writer.Write(row)
		if err != nil {
			log.Fatal("Error: trouble writing to file")
		}
	}
}

// Member variables represent independent (x) and dependent (y) variables
type InputData struct {
	X []float64
	Y []float64
}

// loads in training data from csv file
func LoadTrainingData(filename string) InputData{
	xVector := make([] float64,0)
	yVector := make([] float64,0)
	csvFile, err := os.Open(filename)
	if err != nil {
		log.Fatal("Error: issue with opening csv file")
	}

	csvReader := csv.NewReader(csvFile)
	for {
		line, err := csvReader.Read()
		if err == io.EOF{
			break
		}
		if err != nil {
			log.Fatal("Error: issue with reading line from csv file", line)
		}

		x,_ := strconv.ParseFloat(line[0], 64)
		y,_ := strconv.ParseFloat(line[1], 64)

		xVector = append(xVector, x)
		yVector = append(yVector, y)
	}
	return InputData{xVector, yVector}
}
