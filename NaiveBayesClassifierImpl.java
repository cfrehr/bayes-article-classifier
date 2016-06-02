///////////////////////////////////////////////////////////////////////////////
// Main Class File:	HW5.java
//
// File:       		NaiveBayesClassifierImpl.java
// Description: 	Naive Bayesian classifier of news articles implemented with:
//						- smoothing
//						- consideration of underflow
//
// Run Args:		<modeFlag> : Integer that controls program output
//						0: 	Prints out the number of documents for each label in the training set.
//						1: 	Prints out the number of words for each label in the training set.
//						2: 	For each instance in test set print out a line displaying the predicted
//							class and the log probabilitiesfor both classes.
//						3: 	Prints out the confusion matrix.
//					train.bbc.txt
//					test.bbc.txt
//
// Author:          Cody Frehr
// Course:        	CS 540: Intro to Artificial Intelligence
///////////////////////////////////////////////////////////////////////////////

import java.util.HashMap;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {

	// fields
	int v = 0;
	int sLCount = 0;
	int sWCount = 0;
	int bLCount = 0;
	int bWCount = 0;
	Map<Label,Map<String,Integer>> articleMap = new HashMap<Label,Map<String,Integer>>();


	
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
  	public void train(Instance[] trainingData, int v) {
		
		// initialize fields
		this.v = v;
		HashMap<String,Integer> sWords = new HashMap<String,Integer>();
		HashMap<String,Integer> bWords = new HashMap<String,Integer>();

		// iterate through articles
		int instanceCount = trainingData.length;
		for(int i=0; i<instanceCount; i++) {
			Instance currInst = trainingData[i];
			Label currLabel = currInst.label;
			String[] currWords = currInst.words.clone();
			int wordCount = currWords.length;
			
			// count occurrences of articles types and words in article
			if(currLabel.equals(Label.valueOf("SPORTS"))) {
				sLCount++;
				for(int j=0; j<wordCount; j++) {
					String currWord = currWords[j];
					if(sWords.containsKey(currWord)) {
						sWords.put(currWord, sWords.get(currWord)+1);
					} else {
						sWords.put(currWord,1);
					}
					sWCount++;
				}
			} else {
				bLCount++;
				for(int j=0; j<wordCount; j++) {
					String currWord = currWords[j];
					if(bWords.containsKey(currWord)) {
						bWords.put(currWord, bWords.get(currWord)+1);
					} else {
						bWords.put(currWord,1);
					}
					bWCount++;
				}
			}
		}
		
		// build articleMap
		Label sports = Label.valueOf("SPORTS");
		Label business = Label.valueOf("BUSINESS");
		articleMap.put(sports, sWords);
		articleMap.put(business, bWords);
  	}

	/*
	 * Prints out the number of documents for each label
	 */
	public void documents_per_label_count(){
		
		System.out.println("SPORTS="+sLCount);
		System.out.println("BUSINESS="+bLCount);
	}

	/*
	 * Prints out the number of words for each label
	 */
	public void words_per_label_count(){
		
		System.out.println("SPORTS="+sWCount);
		System.out.println("BUSINESS="+bWCount);	
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		
		double sL = (double) sLCount;
		double bL = (double) bLCount;
		// return probability of an article label
		if(label.equals(Label.valueOf("SPORTS"))) {
			return (sL/(sL+bL));
		} else {
			return (bL/(sL+bL));
		}
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label, i.e. P(word|SPORTS) or
	 * P(word|BUSINESS)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
	
		// initialize conditional probability parameters
		double delta = 0.00001;
		double vSize = (double) v;
		double wordCount = 0.0;
		if(label.equals(Label.valueOf("SPORTS"))) {
			wordCount = (double) sWCount;
		} else {
			wordCount = (double) bWCount;
		}
		double clw = 0.0;
		if(articleMap.get(label).containsKey(word)) {
			clw = (double) articleMap.get(label).get(word);
		}

		// calculate conditional probability of word given label, with smoothing
		double result = 0.0;
		result = (double) ((clw + delta) / ((vSize * delta) + wordCount));
		return result;
		
	}

	/**
	 * Classifies an array of words as either SPAM or HAM.
	 */
	@Override
	public ClassifyResult classify(String[] words) {

		// initialize underflow parameters
		ClassifyResult classification = new ClassifyResult();
		double sLog = 0.0;
		sLog = Math.log(p_l(Label.valueOf("SPORTS")));
		double bLog = 0.0;
		bLog = Math.log(p_l(Label.valueOf("BUSINESS")));
		
		// classify article
		double sLogSum = 0.0;
		double bLogSum = 0.0;
		int wordCount = words.length;
		String currWord = "";
		for(int i=0; i<wordCount; i++) {
			currWord = words[i];
			double sCond = (double) Math.log(p_w_given_l(currWord,Label.valueOf("SPORTS")));
			double bCond = (double) Math.log(p_w_given_l(currWord,Label.valueOf("BUSINESS")));
			sLogSum += sCond;
			bLogSum += bCond;
		}

		classification.log_prob_sports = sLog + sLogSum;
		classification.log_prob_business = bLog + bLogSum;
		if(classification.log_prob_sports >= classification.log_prob_business) {
			classification.label = Label.valueOf("SPORTS");
		} else {
			classification.label = Label.valueOf("BUSINESS");
		}
		
		return classification;
	}
  
	/*
	 * Constructs the confusion matrix
	 */
	@Override
	public ConfusionMatrix calculate_confusion_matrix(Instance[] testData){
		
		Label sports = Label.valueOf("SPORTS");
		Label business = Label.valueOf("BUSINESS");
		
		// get confustion matrix
		ConfusionMatrix matrix = new ConfusionMatrix(0,0,0,0);
		int instSize = testData.length;
		ClassifyResult currClass = null;
		Instance currInst = null;
		for(int i=0; i<instSize; i++) {
			currInst = testData[i];
			currClass = classify(currInst.words);
			if(currInst.label.equals(sports)) {
				if(currClass.label.equals(sports)) {
					matrix.TP++;
				} else {
					matrix.FN++;
				}
			} else {
				if(currClass.label.equals(business)) {
					matrix.TN++;
				} else {
					matrix.FP++;
				}
			}
		}
		
		return matrix;
	}
  
}
