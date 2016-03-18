# CompositionalLM

Source code for Compositional language modeling paper.

## Dependencies:
* Java 1.8
* Maven 3.3 (3.3.9 to be exact)

## Build

We use maven to build the model. Make sure the version of maven is 3.3+. To build the model run

> mvn assembly:assembly

## Train

To train the model, run `train.sh`. 

## Test

To test the model, run `test.sh`

### Configuration

Training, testing configuration are in configs/ folder. The config folder contains six config files. The important configs in these files are:

* config/train.config
    >This the the most important and longest config file as it deals with training configurations. The important properties to notice here are:
  
 	- maxOptimizerEpochs: numer of M step optimization epochs.
 	- maxEMEpochs: number of EM epochs.
 	- batchSize 
 	- validationBatchSize
 	- learningRate
 	- l2term: l2 regularization coefficient.

* config/model.config
	> Model config contains model related configuration. Most important of which is the size of the latent space.
	
	- dimensions: The dimension of the latent space.
	- outFile: The place to save the model file after training.
	- inFile: Input model file while testing. 
* config/test.config
	> This controls the testing configuration. Important thing to note while testing is to ensure that the dimension in model config is same as the one used while training. 

	- testBatchSize: Size of the test batch.