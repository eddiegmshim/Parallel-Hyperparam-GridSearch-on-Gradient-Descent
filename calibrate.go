package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"proj3/data"
	"proj3/regression"
	"runtime"
	"strconv"
	"sync"
)

// Instructions for input args
func printUsage() {
	usage := "calibrate -p=number of threads -g=sample size -i=\"filename.csv\" -b=block size < inputHyperparams.txt\n" +
		"\t-t=number of threads = An optional flag to run the editor in its parallel version.\n" +
		"\t-g=sample size = An optional flag to generate data of size n.\n" +
		"\t-i=\"filename.csv\" = filepath of cached input data csv file\n" +
		"\t-b=block size = block size, defined as number of JSON tasks a reader should attempt to chunk and grab\n" +
		"\t inputHyperparams = JSON text file of hyperparameters we want to test"
	fmt.Printf("Incorrect input commands, -f flag is required. Please use following commands:\n" + usage)
}

func main(){
	inpath := flag.String("i", "", "filepath string")
	numThreads := flag.Int("t", 0, "an int representing number of threads")
	generateData := flag.Int("g", 0, "an int representing size of sample data to generate")
	blockSize := flag.Int("b", 1, "number of JSON tasks a reader should attempt to chunk and grab")
	flag.Parse()
	fmt.Println("Input args:", "-t:", *numThreads, "| -g:", *generateData, "| -i:", *inpath, "| -b:", *blockSize)
	if *generateData != 0 && *inpath != "" { //only generate data or run gradient descent, not both
		printUsage()
		os.Exit(0)
	}

	var trainingData data.InputData
	if *generateData != 0 {
		*inpath = "trainingData_" + strconv.Itoa(*generateData) + ".csv"
		data.GenerateTrainingData(*generateData, *inpath)
		fmt.Println("Generated training data into filepath:", *inpath)
		os.Exit(0)
	} else {
		trainingData = data.LoadTrainingData(*inpath)
	}

	if *numThreads == 0 {
		gridSearchSequential(trainingData)
	} else {
		gridSearchParallel(trainingData, *numThreads, *blockSize)
	}
}

func gridSearchSequential(data data.InputData){
	minX, maxX := regression.MinMax(data.X)
	dataNormalized := regression.Normalize(data, minX, maxX)
	hyperParamsTasks := readJSONInputTasks()
	optimalHyperParamsArr := make([]Hyperparameters, 0)
	optimalModelParamsArr := make([]regression.Parameters,0)

	for _, hyperParams := range hyperParamsTasks {
		optimalHyperParams := Hyperparameters{hyperParams.Outpath,nil, nil, nil, nil}
		optimalMSE := math.MaxFloat64
		optimalModelParams := regression.Parameters{0, 0}

		for _,alpha := range hyperParams.Alpha {
			for _, numEpochs := range hyperParams.NumEpochs {
				parameters := runGradientDescent(dataNormalized, alpha, numEpochs)
				parameters = regression.UnNormalize(parameters, data, minX, maxX)

				predicted := regression.Forecast(parameters.Mu, parameters.Beta, data.X)
				mse := regression.CalcMSE(predicted, data.Y)
				if mse < optimalMSE{
					optimalMSE = mse
					optimalHyperParams = Hyperparameters{hyperParams.Outpath, []float64{alpha}, []float64{numEpochs}, nil, nil}
					optimalModelParams = regression.Parameters{parameters.Mu, parameters.Beta}
				}
			}
		}
		optimalHyperParamsArr = append(optimalHyperParamsArr, optimalHyperParams)
		optimalModelParamsArr = append(optimalModelParamsArr, optimalModelParams)
		writer(optimalHyperParams, optimalModelParams, nil)
	}
}

// Top level of grid search parallel
func gridSearchParallel(data data.InputData, numThreads int, blockSize int) {
	runtime.GOMAXPROCS(numThreads)
	numReaders := int(math.Ceil(float64(numThreads) * (1.0/5.0)))
	readerDone := make(chan bool)

	var readerMutex sync.Mutex // a lock to allow us to have multiple threads read from Stdin in thread safe manner
	dec := json.NewDecoder(os.Stdin)

	for i := 0; i < numReaders; i++ {
		go reader(data, numThreads, blockSize, readerDone, &readerMutex, dec)
	}

	//wait until all readers are done using a channel
	for i := 0; i < numReaders; i++{
		<- readerDone
	}
}

// A goroutine that reads Stdin JSON tasks in parallel
func reader(data data.InputData, numThreads int, blockSize int, readerDone chan bool, mutex *sync.Mutex, dec *json.Decoder){
	for true {
		hyperparamsTaskChannel := readJSONInputTasksParallel(mutex, blockSize, dec)
		numTasks := len(hyperparamsTaskChannel)
		if numTasks == 0 {
			readerDone <- true
			break
		}

		//every reader spawns a single worker pipeline goroutine
		workerDone := make(chan bool, 1)
		go worker(data, numThreads, numTasks, hyperparamsTaskChannel, workerDone)
		close(hyperparamsTaskChannel) //close out the imageTasksChannel once worker is done processing it

		//wait until worker goroutine finishes
		<- workerDone
	}
}

// A goroutine which takes in a grid of hyperparameters, and splits it into chunks we can work on in parallel
func worker(data data.InputData, numThreads int, numTasks int, hyperparamsTaskChannel <- chan Hyperparameters, workerDone chan bool) {
	minX, maxX := regression.MinMax(data.X)
	dataNormalized := regression.Normalize(data, minX, maxX)
	globalOptimalHyperParamsArr := make([]Hyperparameters, 0)
	globalOptimalModelParamsArr := make([]regression.Parameters,0)

	for taskCounter := 0; taskCounter < numTasks; taskCounter++{ // loop through each hyperParam set in within our numTasks each reader is responsible for
		hyperParams := <- hyperparamsTaskChannel
		globalOptimalHyperParams := &Hyperparameters{ hyperParams.Outpath, nil, nil, nil, nil}
		globalOptimalMSE := new(float64)
		*globalOptimalMSE = math.MaxFloat64
		globalOptimalModelParams := &regression.Parameters{0, 0}

		numTotalParamSets := math.Max(1, float64(len(hyperParams.Alpha))) * math.Max(1, float64(len(hyperParams.NumEpochs))) *
			math.Max(1, float64(len(hyperParams.Lambda))) * math.Max(1, float64(len(hyperParams.MiniBatchSize)))
		workSizePerThread := math.Ceil(numTotalParamSets / float64(numThreads))
		workArray := createArrayParamPermutations(hyperParams)
		var group sync.WaitGroup
		var globalParamLock sync.Mutex

		for i := 0; i < numThreads; i++ {
			startIndex := float64(i) * workSizePerThread
			endIndex := float64(i + 1) * workSizePerThread
			if endIndex > float64(len(workArray)){
				break
			}
			group.Add(1)
			subworkArray :=  workArray[int(startIndex) : int(endIndex)]
			go runParallelGradientDescent(dataNormalized, data, minX, maxX, &group, &globalParamLock, subworkArray, globalOptimalHyperParams,
				globalOptimalMSE, globalOptimalModelParams)

		}
		group.Wait()
		globalOptimalHyperParamsArr = append(globalOptimalHyperParamsArr, *globalOptimalHyperParams)
		globalOptimalModelParamsArr = append(globalOptimalModelParamsArr, *globalOptimalModelParams)

		//write results
		writerDone := make(chan bool, 1)
		go writer(*globalOptimalHyperParams, *globalOptimalModelParams, writerDone)
		<- writerDone //wait until writer goroutine finishes
	}

	//finished with worker
	workerDone <- true
}

// A goroutine which writes our final hyperparameters into an output csv file
func writer(globalOptimalHyperParams Hyperparameters, globalOptimalModelParams regression.Parameters, writerDone chan bool) {
	file, err := os.Create(globalOptimalHyperParams.Outpath)
	if err != nil {
		log.Fatal("Error: cannot create output file", err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"alpha", "numEpochs", "lambda", "miniBatchSize", "beta", "mu"}
	writer.Write(header)
	alphaWrite, numEpochsWrite, lambdaWrite, miniBatchSizeWrite := "", "", "", ""
	if globalOptimalHyperParams.Alpha != nil{
		alphaWrite = fmt.Sprintf("%f", globalOptimalHyperParams.Alpha[0])
	} else {
		alphaWrite = "NA"
	}
	if globalOptimalHyperParams.NumEpochs != nil{
		numEpochsWrite = fmt.Sprintf("%f", globalOptimalHyperParams.NumEpochs[0])
	} else {
		numEpochsWrite = "NA"
	}
	if globalOptimalHyperParams.Lambda != nil{
		lambdaWrite = fmt.Sprintf("%f", globalOptimalHyperParams.Lambda[0])
	} else {
		lambdaWrite = "NA"
	}
	if globalOptimalHyperParams.MiniBatchSize != nil{
		miniBatchSizeWrite = fmt.Sprintf("%f", globalOptimalHyperParams.MiniBatchSize[0])
	} else {
		miniBatchSizeWrite = "NA"
	}

	betaWrite := fmt.Sprintf("%f", globalOptimalModelParams.Beta)
	muWrite := fmt.Sprintf("%f", globalOptimalModelParams.Mu)
	stringHyperparam := []string{alphaWrite, numEpochsWrite, lambdaWrite, miniBatchSizeWrite, betaWrite, muWrite}
	fmt.Println(stringHyperparam)
	err = writer.Write(stringHyperparam)
	if err != nil {
		fmt.Println("error:")
		log.Fatal("Error: cannot write hyperparam into file")
	}

	//writerDone is nil in sequential version, else exists in parallel version
	if writerDone != nil {
		writerDone <- true
	}
}

// Generates an array of all permuations of hyperparmeters, given a grid of hyperparameters
func createArrayParamPermutations (hyperparameters Hyperparameters) [] Hyperparameters{
	output := make([]Hyperparameters, 0, 0)
	for _, alpha := range hyperparameters.Alpha {
		for _, numEpochs := range hyperparameters.NumEpochs {
			permutation := Hyperparameters{hyperparameters.Outpath, []float64{alpha}, []float64{numEpochs}, nil, nil}
			output = append(output, permutation)
		}
	}
	return output
}

// Calibrates regression coefficients using gradient descent
func runGradientDescent(dataNormalized data.InputData, alpha float64, numEpochs float64) regression.Parameters{
	parameters := regression.Parameters{0,0} //at the start of gradient descent, initialize all params =0
	for i:=0; i < int(numEpochs); i++{
		parameters = regression.UpdateParams(parameters, dataNormalized, alpha)
	}
	return parameters
}

// Calibrates global optimal hyperparameters in parallel using gradient descent
func runParallelGradientDescent(dataNormalized data.InputData, data data.InputData, minX float64, maxX float64,
	group *sync.WaitGroup, globalParamLock *sync.Mutex, workArray []Hyperparameters,
	globalOptimalHyperParams *Hyperparameters, globalOptimalMSE *float64, globalOptimalModelParams *regression.Parameters) {

	localOptimalHyperParams := Hyperparameters{globalOptimalHyperParams.Outpath, nil, nil, nil, nil}
	localOptimalMSE := math.MaxFloat64
	localOptimalModelParams := regression.Parameters{0, 0}

	for _, hyperParams := range workArray {
		parameters := runGradientDescent(dataNormalized, hyperParams.Alpha[0], hyperParams.NumEpochs[0])
		parameters = regression.UnNormalize(parameters, data, minX, maxX)

		predicted := regression.Forecast(parameters.Mu, parameters.Beta, data.X)
		mse := regression.CalcMSE(predicted, data.Y)
		if mse < localOptimalMSE {
			localOptimalMSE = mse
			localOptimalHyperParams = Hyperparameters{globalOptimalHyperParams.Outpath, []float64{hyperParams.Alpha[0]}, []float64{hyperParams.NumEpochs[0]}, nil, nil}
			localOptimalModelParams = regression.Parameters{parameters.Mu, parameters.Beta}
		}
	}
	if localOptimalMSE < *globalOptimalMSE {
		globalParamLock.Lock()
		*globalOptimalMSE = localOptimalMSE
		*globalOptimalHyperParams = localOptimalHyperParams
		*globalOptimalModelParams = localOptimalModelParams
		globalParamLock.Unlock()
	}
	group.Done()
}

// Reads in Stdin JSON inputs sequentially
func readJSONInputTasks() []Hyperparameters{
	var hyperParams []Hyperparameters
	dec := json.NewDecoder(os.Stdin)
	for { //loop through and process each json object as task
		var j jsonInput
		var h Hyperparameters
		err := dec.Decode(&j)
		if err != nil {
			if err == io.EOF{
				break
			}
			fmt.Println(err)
		}
		h.Outpath = j.Outpath
		h.Alpha = stringToFloat64(j.Alpha)
		h.NumEpochs = stringToFloat64(j.NumEpochs)
		h.Lambda = stringToFloat64(j.Lambda)
		h.MiniBatchSize = stringToFloat64(j.MiniBatchSize)
		hyperParams = append(hyperParams, h)
	}
	return hyperParams
}

// Reads in Stdin JSON inputs in a thread safe manner by locking each time it's called. Reader goroutines will
// all attempt to access Stdin through this function. Outputs a channel of Hyperparameter tasks that gets passed downstream to
// worker goroutine
func readJSONInputTasksParallel(lock *sync.Mutex, blockSize int, dec *json.Decoder) chan Hyperparameters{
	lock.Lock()
	hyperparamsTasksChannel := make(chan Hyperparameters, blockSize)
	for i:=0; i < blockSize; i++{ //loop through blocksize amount of each json objects as ImageTask
		var j jsonInput
		var h Hyperparameters
		err := dec.Decode(&j)
		if err != nil {
			if err == io.EOF{
				break
			}
		}
		h.Outpath = j.Outpath
		h.Alpha = stringToFloat64(j.Alpha)
		h.NumEpochs = stringToFloat64(j.NumEpochs)
		h.Lambda = stringToFloat64(j.Lambda)
		h.MiniBatchSize = stringToFloat64(j.MiniBatchSize)
		hyperparamsTasksChannel <- h
	}
	lock.Unlock()
	return hyperparamsTasksChannel
}

// Each line from Stdin represents a JSON task which has the hyperparameters we want to test
type jsonInput struct {
	Outpath string `json:"outpath"`
	Alpha []string `json:"alpha"`
	NumEpochs []string `json:"numEpochs"`
	Lambda []string `json:"lambda"`
	MiniBatchSize []string `json:"miniBatchSize"`
}

// Converted jsonInput into float64 vars
type Hyperparameters struct {
	Outpath string
	Alpha []float64
	NumEpochs []float64
	Lambda []float64
	MiniBatchSize []float64
}

func stringToFloat64(input []string) []float64{
	output := make([]float64, 0)
	for i:=0; i<len(input); i++{
		conv, _ := strconv.ParseFloat(input[i], 64)
		output = append(output, conv)
	}
	return output
}