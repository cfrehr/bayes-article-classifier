///////////////////////////////////////////////////////////////////////////////
// Main Class File:	HW5.java
//
// File:       		NaiveBayesClassifier.java
// Author:          Chuck Dyer
// Course:        	CS 540: Intro to Artificial Intelligence
///////////////////////////////////////////////////////////////////////////////

/**
 * Interface class for the naive bayes classifier. For an explanation of methods, see
 * NaiveBayesClassifierImpl.
 * 
 * DO NOT MODIFY
 */
public interface NaiveBayesClassifier {
  void train(Instance[] trainingData, int v);

  double p_l(Label label);

  double p_w_given_l(String word, Label label);

  public ClassifyResult classify(String[] words);
  
  public ConfusionMatrix calculate_confusion_matrix(Instance[] testData);

  public void words_per_label_count();

  public void documents_per_label_count();
}
