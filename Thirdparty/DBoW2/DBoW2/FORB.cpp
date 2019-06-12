/**
 * File: FORB.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 * Distance function has been modified 
 *
 */

 
#include <vector>
#include <string>
#include <sstream>
#include <stdint-gcc.h>

#include "FORB.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

// the length of the orb descriptor
const int FORB::L=32;

// typedef cv::Mat TDescriptor;
// typedef const TDescriptor *pDescriptor;
void FORB::meanValue(const std::vector<FORB::pDescriptor> &descriptors, 
  FORB::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    mean.release();
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    // the size of the vector is FORB::L*8
    vector<int> sum(FORB::L * 8, 0);
    
    for(size_t i = 0; i < descriptors.size(); ++i)// for all descriptors in the vector
    {
      const cv::Mat &d = *descriptors[i];//get the current descriptor
      const unsigned char *p = d.ptr<unsigned char>(); // get the current descriptor as a container of uchar type
      
      for(int j = 0; j < d.cols; ++j, ++p) //for every char in the current descriptor, every char has 8 bits, so the vector sum is 8 times the size of the descriptor
      {
        if(*p & (1 << 7)) ++sum[ j*8     ];// if the highest bit in the char is not 0, the corresponding int in the vector sum increase by 1
        if(*p & (1 << 6)) ++sum[ j*8 + 1 ];// if the second bit in the char is not 0, the corresponding int in the vector sum increase by 1
        if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
        if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
        if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
        if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
        if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
        if(*p & (1))      ++sum[ j*8 + 7 ];
      }
    }
    
    mean = cv::Mat::zeros(1, FORB::L, CV_8U);// the mean is a mat whose size is (1, FORB::L),and the type of each element is uchar
    unsigned char *p = mean.ptr<unsigned char>(); // p is a pointer of type uchar
    
    const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2; //half the number of the descriptors in the vector
    for(size_t i = 0; i < sum.size(); ++i)
    {
      if(sum[i] >= N2)// sum[i]>=N2 means sum[i]/descriptor.size()=1, i.e. the corresponding bit in mean is 1 not 0
      {
        // set bit
        *p |= 1 << (7 - (i % 8));
      }
      
      if(i % 8 == 7) ++p;// if have processed all the 8 elements representing the whole uchar, move the pointer to the next element in cv::Mat mean
    }
  }
}

// --------------------------------------------------------------------------
  
/*
对于v = v - ((v >> 1) & 0x55555555);
记v为                                    a1  a2  a3  a4    a5  a6  a7  a8    a9  a10 a11 a12   a13 a14 a15 a16   a17 a18 a19 a20   a21 a22 a23 a24   a25 a26 a27 a28   a29 a30 a31 a32
则对v右移又与上8位的16进制5,即8位的0101为   0  a1   0  a3     0  a5   0  a7     0   a9  0  a11    0  a13  0  a15    0  a17  0  a19    0  a21  0  a23    0  a25  0  a27    0  a29  0  a31
对于相邻的两位 a1a2-0a1的结果为   
1）若a1=1,a2=1,则0a1为01,11-01为10即代表a1a2有两个1
2）若a1=1,a2=0,则0a1为01,10-01为01即代表a1a2有一个1
3）若a1=0,a2=1,则0a1为00,01-00为01即代表a1a2有一个1
4）若a1=0,a2=0,则0a1为00,00-00为00即代表a1a2没有1
这样进行减法后，相邻两位的数值即为相邻两位中1的个数
a1  a2  a3  a4    a5  a6  a7  a8    a9  a10 a11 a12   a13 a14 a15 a16   a17 a18 a19 a20   a21 a22 a23 a24   a25 a26 a27 a28   a29 a30 a31 a32
 0  a1   0  a3     0  a5   0  a7     0   a9  0  a11    0  a13  0  a15    0  a17  0  a19    0  a21  0  a23    0  a25  0  a27    0  a29  0  a31
 \  /    \  /      \  /    \  /      \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /
a1a2    a3a4       a5a6    a7a8       a9a10   a11a12    .....
1的个数 1的个数    1的个数  1的个数     1的个数  1的个数   .....
进行上面这步以后，二进制串中的值已经从描述子不同的位数变成了相邻位中1的个数
记suma_b表示a_b两位中1的个数，则最新的v变为
sum1_2  sum3_4  sum5_6  sum7_8  sum9_10  sum11_12  sum13_14  sum15_16  sum17_18  sum19_20  sum21_22  sum23_24  sum25_26  sum27_28  summ29_30  sum31_32
v = (v & 0x33333333) + ((v >> 2) & 0x33333333)分析如下
v按位与8个16进制3即0011为
  0  0  sum3_4    0  0  sum7_8    0  0   sum11_12     0   0  sum15_16    0   0   sum19_20    0   0   sum23_24    0   0   sum27_28     0   0   sum31_32  
v右移两位再按位与8个16进制3即0011为
  0  0  sum1_2    0  0  sum5_6    0  0   sum9_10      0   0  sum13_14    0   0   sum17_18    0   0   sum21_22    0   0   sum25_26     0   0   sum29_30
二者再相加以后，v的值已经变成
sum1_2_3_4   sum5_6_7_8   sum9_10_11_12   sum13_14_15_16   sum17_18_19_20   sum21_22_23_24   sum25_26_27_28   sum29_30_31_32
8组二进制中最大的值为4 ，两组相加不超过8，还是可以存在4位二进制数中不用进位
(((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24分析如下
v右移四位为
0  0  0  0    sum1_2_3_4   sum5_6_7_8   sum9_10_11_12   sum13_14_15_16   sum17_18_19_20   sum21_22_23_24   sum25_26_27_28  
v加上v右移四位为
sum1_2_3_4   sum1_2_3_4_5_6_7_8   sum5_6_7_8_9_10_11_12   sum9_10_11_12_13_14_15_16   sum13_14_15_16_17_18_19_20   sum17_18_19_20_21_22_23_24   sum21_22_23_24_25_26_27_28   sum25_26_27_28_29_30_31_32
这个值再与上16进制F0F0F0F即为   0000 1111 0000 1111 0000 1111 0000 1111为
0000   sum1_2_3_4_5_6_7_8   0000   sum9_10_11_12_13_14_15_16   0000   sum17_18_19_20_21_22_23_24   0000   sum25_26_27_28_29_30_31_32
再乘以1010101即为
0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                               0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                                                                              0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                                                                                                                            0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
dist为32位2进制，1位16进制为4位2进制，32位2进制代表8位16进制,故而上面结果中前6位16进制溢出了,剩下的结果为
 0000         sum25_26_27_28_29_30_31_32
    0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
     0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
    0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
此时最大可能值为32,需要6位二进制两位16进制表示，故而下列结果将两位16进制写在一起，加法结果为
sum1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum25_26_27_28_29_30_31_32
向右移24位可得sum1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32 即为二进制串中1的个数
*/
// first， apply the xor operation of the two descriptors, if one of the corresponding bit is 1 and the other bit is 0, the bit is set to 1 in the result, then count the 
// number of 1 in the result , which is the hamming distance of the two descriptors
int FORB::distance(const FORB::TDescriptor &a,
  const FORB::TDescriptor &b)
{
  // Bit set count operation from
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist=0;

  for(int i=0; i<8; i++, pa++, pb++)
  {
      unsigned  int v = *pa ^ *pb;
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

// --------------------------------------------------------------------------
// convert the descriptor to string
std::string FORB::toString(const FORB::TDescriptor &a)
{
  stringstream ss;
  const unsigned char *p = a.ptr<unsigned char>();
  
  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << (int)*p << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
// get the descriptor from a string 
void FORB::fromString(FORB::TDescriptor &a, const std::string &s)
{
  a.create(1, FORB::L, CV_8U);
  unsigned char *p = a.ptr<unsigned char>();
  
  stringstream ss(s);
  for(int i = 0; i < FORB::L; ++i, ++p)
  {
    int n;
    ss >> n;
    
    if(!ss.fail()) 
      *p = (unsigned char)n;
  }
  
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const size_t N = descriptors.size();
  
  mat.create(N, FORB::L*8, CV_32F);
  float *p = mat.ptr<float>();
  
  for(size_t i = 0; i < N; ++i)
  {
    const int C = descriptors[i].cols;
    const unsigned char *desc = descriptors[i].ptr<unsigned char>();
    
    for(int j = 0; j < C; ++j, p += 8)
    {
      p[0] = (desc[j] & (1 << 7) ? 1 : 0);
      p[1] = (desc[j] & (1 << 6) ? 1 : 0);
      p[2] = (desc[j] & (1 << 5) ? 1 : 0);
      p[3] = (desc[j] & (1 << 4) ? 1 : 0);
      p[4] = (desc[j] & (1 << 3) ? 1 : 0);
      p[5] = (desc[j] & (1 << 2) ? 1 : 0);
      p[6] = (desc[j] & (1 << 1) ? 1 : 0);
      p[7] = desc[j] & (1);
    }
  } 
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  mat.create(descriptors.size(), 32, CV_8U);
  
  unsigned char *p = mat.ptr<unsigned char>();
  
  for(size_t i = 0; i < descriptors.size(); ++i, p += 32)
  {
    const unsigned char *d = descriptors[i].ptr<unsigned char>();
    std::copy(d, d+32, p);
  }
  
}

// --------------------------------------------------------------------------

} // namespace DBoW2


