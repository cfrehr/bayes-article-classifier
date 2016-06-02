
NAIVE BAYESIAN CLASSIFIER FOR NEWS ARTICLE IDENTIFICATION
---------------------------------------------------------

I. ABOUT
--------
This is a naive bayesian classifier that classifies news articles based on their word content. 
Data was pulled from http://mlg.ucd.ie/datasets/bbc.html

This classifier was implemented using:
	- smoothing
	- consideration of underflow


II. RUN ARGS
------------
Run args:
	<modeFlag> | Integer that controls program output
		"0" :	Prints out the number of documents for each label in the training set.
		"1" :	Prints out the number of words for each label in the training set.
		"2" :	For each instance in test set print out a line displaying the predicted
			class and the log probabilitiesfor both classes.
		"3" :	Prints out the confusion matrix.
	<train>    | Training data
		"train.bbc.txt"
	<test>     | Testing data					
		"test.bbc.txt"


III. FILES
----------
NaiveBayesClassifierImpl.java     	(Cody Frehr)
ClassifyResult.java			(Chuck Dyer)
ConfusionMatrix.java			(Chuck Dyer)
Instance.java				(Chuck Dyer)
Label.java				(Chuck Dyer)
NaiveBayesClassifier.java		(Chuck Dyer)
