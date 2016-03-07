package com.kushalarora.compositionalLM.lang;

/**
 * Created by arorak on 12/12/15.
 */
public interface IInsideOutsideScoreCalculator
{
    public void doLexScores(final IInsideOutsideScore score);
    public void doInsideScores(final IInsideOutsideScore score);
    public void doOutsideScores(final IInsideOutsideScore score);
    public void doMuScore(final IInsideOutsideScore score);
    public IInsideOutsideScore getScore(final Sentence sentence);
    public void computeInsideOutsideProb(final IInsideOutsideScore score);
}
