/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{
// const ORBVocabulary* mpVoc; associated vocabulary
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    // std::vector<list<KeyFrame*> > mvInvertedFile; inverted file
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)//DBoW2::BowVector mBowVec
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);//若关键帧中存在某词汇，则将该关键帧存入该词汇对应的mvInvertedFile中
}

void KeyFrameDatabase::erase(KeyFrame* pKF)//将某点从其对应的词汇的mvInvertedFile中删除
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}


vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();//获取与某关键帧相关联的所有关键帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes 获取与当前关键帧共有某个词汇的所有关键帧
    // Discard keyframes connected to the query keyframe 丢弃与有疑问关键帧关联的关键帧
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

/*
 * 1. get all keyframes that share words with F and compute how many words the two frame share, if there is no keyframe shares words with F, return an 
 *    empty vector
 * 2. for every keyframe that shares word with F, find the max number of common words, compute the min number of common words as 0.8 times max number
 *    of common words
 * 3. for every keyframe that shares word with F, if the keyframe shares more than threshold word with F, compute the score between bow vector of the 
 *    two frame, set the relocScore of the keyframe, make the score and the keyframe a pair and push them into a list, if the vector of pair is 
 *    empty, return an empty vector
 * 4. for every pair in the list lScoreAndMatch, get the keyframe and the best covisible keyframes of the keyframe, accumulate score among covisible 
 *    keyframes and make pair of the accscore and the covisible keyframe of the best score and put the pair into lAccScoreAndMatch, find the best 
 *    accScore among all the lScoreAndMatch
 * 5. compute the accumulate score threshold as 0.75 times the best accumulate score
 * 6. for every pair of the accumulated score and the best keyframe among covisible frames in the list lAccScoreAndMatch, if the accumulated score is 
 *    bigger than the threshold and the keyframe haven't been put into the set before, push the keyframe to vpRelocCandidates and to the set
 * 7. return vpRelocCandidates
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;// keyframes that shares words with frame F

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        // class BowVector: public std::map<WordId, WordValue>
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)// for every word in the bow vector of F
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];// use the bow word id to get keyframes

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)// for every keyframe in the list
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)// if the mnRelocQuery of pKFi is not the frameid of F
                {
                    pKFi->mnRelocWords=0;// mnRelocWords means how many words the pKFi and its reloc query share
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty()) // if there is no keyframe shares word with F, return an empty vector 
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    //for every keyframe that shares word with F,find the one with the max number of common words
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;// compute the min number of common words using the max number of common words

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    /* the class KeyFrame has some variables corresponding to relocalization, for example
     * 1. long unsigned int mnRelocQuery
     * 2. int mnRelocWords
     * 3. float mRelocScore
     */
    // Compute similarity score. for every keyframe that shares word with F
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords) // if the keyframe shares more than threshold word with F
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);// compute the score between two bow vector
            pKFi->mRelocScore=si;// set the relocScore of the keyframe
            lScoreAndMatch.push_back(make_pair(si,pKFi));// make the score and the keyframe a pair and push them into a list
        }
    }

    if(lScoreAndMatch.empty())// if the list containing score and keyframe is empty, return an empty vector
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        /* vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) if the keyframe has more than N ordered connected keyframes, get the first
           N keyframes in the vector, else get the whole vector*/ 
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        // accScore means the sum of the score between F and some covisible keyframes
        // for every keyframe in the vpNeighs got above
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;// get the neighbor keyframe
            if(pKF2->mnRelocQuery!=F->mnId)// if the neighbor keyframe's relocQuery is not F, continue to process next neighbor keyframe
                continue;

            accScore+=pKF2->mRelocScore;// add the relocScore of the neighbor keyframe to accScore
            if(pKF2->mRelocScore>bestScore)// if the relocScore of the neighbor keyframe is bigger than the bestScore
            {
                pBestKF=pKF2;//set best keyframe to be the neighbor keyframe
                bestScore = pKF2->mRelocScore;// set bestScore tobe the relocScore of the neighbor keyframe
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));// make pair of the accumulated score and the best keyframe among covisible frames
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;// compute the accumulate score threshold as 0.75 miltiply the best accumulate score
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    // for every pair of the accumulated score and the best keyframe among covisible frames in the list lAccScoreAndMatch
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;// get the accumulated score
        if(si>minScoreToRetain)// if the accumulated score is bigger than the threshold
        {
            KeyFrame* pKFi = it->second;//get the keyframe
            if(!spAlreadyAddedKF.count(pKFi))// if the keyframe haven't been put into the set
            {
                vpRelocCandidates.push_back(pKFi);// push the keyframe to vector vpRelocCandidates
                spAlreadyAddedKF.insert(pKFi);// insert the keyframe to set spAlreadyAddedKF
            }
        }
    }

    return vpRelocCandidates;// return the vector of keyframes vpRelocCandidates
}

} //namespace ORB_SLAM
