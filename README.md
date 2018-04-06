# regression_model


usage: train.py [-h]

--s S 

--trainfile TRAINFILE [--delimiter DELIMITER]
                
[--modelfile MODELFILE] 

[--method METHOD]
                
[--iterations ITERATIONS] 

[--alpha ALPHA]

[--threshold THRESHOLD]

optional arguments:

  -h, --help            show this help message and exit
  
  --s S                 0 linearRegression, 1 logisticRegression
  
  --trainfile TRAINFILE
                        The path of the training file
                        
  --delimiter DELIMITER
                        Delimiters for training files,the default is a space
                        
  --modelfile MODELFILE
                        Output model file
                        
  --method METHOD       0 gradientAscent(or gradientDescent) , 1
                        stocGradientAscent(or stocGradientDescent)
                        
  --iterations ITERATIONS
                        Number of iterations of training
                        
  --alpha ALPHA         learning rate
  
  --threshold THRESHOLD
                        Threshold of model classification
                        
example one:

python train.py --s 0 --trainfile trainfile\train_linear.txt --modelfile modelfile\model_linear.txt

output:

theta:

[[1.77179994]
 [2.87766185]]
 
cost:

3.2566489362105466

![Alt text](https://github.com/2014214128/regression_model/raw/master/pic/1.jpg)

example two:

python train.py --s 1 --trainfile trainfile\train_logistic.txt --modelfile modelfile\model_logistic.txt

theta:

[[13.29983403]
 [ 1.15405399]
 [-1.8092296 ]]
 
cost:

[[0.09376224]]

Train Accuracy: 95.000000%

![Alt text](https://github.com/2014214128/regression_model/raw/master/pic/2.jpg)




usage: predict.py [-h] 

--s S 

--modelfile MODELFILE 

--predictfile PREDICTFILE

[--delimiter DELIMITER] 
             
--resultfile RESULTFILE

[--threshold THRESHOLD]

optional arguments:

  -h, --help            show this help message and exit
  
  --s S                 0 linearRegression, 1 logisticRegression
  
  --modelfile MODELFILE
                        The trained model file
                        
  --predictfile PREDICTFILE
                        The path of the training file
                        
  --delimiter DELIMITER
                        Delimiters for predict files,the default is a space
                        
  --resultfile RESULTFILE
                        Forecast result document
                        
  --threshold THRESHOLD
                        Threshold of model classification
                        

example three:

python predict.py --s 0 --modelfile modelfile\model_linear.txt --predictfile predictfile\predict_linear.txt --resultfile resultfile\result_linear.txt

![Alt text](https://github.com/2014214128/regression_model/raw/master/pic/3.jpg)

example four:

python predict.py --s 1 --modelfile modelfile\model_logistic.txt --predictfile predictfile\predict_logistic.txt --resultfile resultfile\result_logistic.txt

![Alt text](https://github.com/2014214128/regression_model/raw/master/pic/4.jpg)
