///////////////////////////////////////////////////////////////////////////////
// Main Class File:	HW5.java
//
// File:       		ClassifyResult.java
// Author:          Chuck Dyer
// Course:        	CS 540: Intro to Artificial Intelligence
///////////////////////////////////////////////////////////////////////////////

/**
 * An SMS instance.
 * 
 * DO NOT MODIFY.
 */
public class ClassifyResult {
  /**
   * Spam or ham
   */
  public Label label;
  /**
   * The log probability that the news article is about sports Does not have to be normalized
   */
  public double log_prob_sports;
  /**
   * The log probability that the news article is about business Does not have to be normalized
   */
  public double log_prob_business;
}
