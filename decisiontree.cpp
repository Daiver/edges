#include "defines.h"
#include "decisiontree.h"
#include "dimSort.h"

#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
//#include <unordered_set>
#include <set>
#include <opencv2/core/core.hpp>

#ifdef ENABLE_TBB_NODES
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_group.h"
#endif

double ginii2(int *freqs, int num_of_classes, int num_of_samples){
    //int* freqs = this->getFreq(labels, num_of_classes);
    double imp = 0;
    //printf("ns %d\n", num_of_samples);
    for(int i = 0; i < num_of_classes; i++){
        double p1 = freqs[i]/(double)num_of_samples;
        for(int j = 0; j < num_of_classes; j++){
            if (i == j) continue;
            double p2 = freqs[j]/(double)num_of_samples;
            imp += p1*p2;
        }
    }
    return imp;
}


void DecisionTree::train(
        const std::vector<InputData> *data, 
        std::vector<int> &data_idx,
        std::vector<cv::Mat> segments){
    //this->num_of_classes = getNumOfClasses(labels);
    //printf("HERe\n");
    //sleep(10);
    //this->calcUniqValues(data);
    this->train_data = data;
    //std::vector<int> data_idx;
    /*for(int i = 0; i < data->size(); i++)
        data_idx.push_back(i);*/
    this->head = buildnode(data_idx, segments, 0);
}

cv::Mat DecisionTree::predict(InputData data){
    TreeNode *node = this->head;
    while(node->type != 2){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        if(data[b->col] >= b->value)
            node = b->right;
        else
            node = b->left;
    }
    return static_cast<TreeLeaf*>(node)->patch;
    //return static_cast<TreeLeaf*>(node)->freqs;
}

/*void DecisionTree::calcUniqValues(const std::vector<InputData> *data){
    this->uvalues = new std::set<InputValue>[data->at(0).size()];
    for(int i = 0; i < data->at(0).size(); i++){
        for(int j = 0; j < data->size(); j++){
            //uvalues[i].insert(round(data->at(j)[i]));
            uvalues[i].insert(data->at(j)[i]);
        }
    }
    for(int i = 0; i < data->at(0).size(); i++){
        if(uvalues[i].size() > 5000)
            printf("USIZE %d\n", uvalues[i].size());
    }*/
    /*printf("FIN\n");
    sleep(10);
    printf("FIN\n");*/
    /*for(int i = 0; i < data[0].size(); i++){
        for(auto val : uvalues[i]){
            printf("%d\n", val);
    printf("FIN\n");
        }
    }*/
    //std::vector<InputValue>* res = new std::vector[this->input_length];
//}

void DecisionTree::divideSet(
        //const std::vector<InputData> &data, 
        const std::vector<int> &data_idx, 
        const std::vector<OutputData> &labels,
        //const std::vector<cv::Mat> &seg,
        int col, InputValue value, 
        std::vector<OutputData> *l1, std::vector<OutputData> *l2,
        //std::vector<cv::Mat> *g1, std::vector<cv::Mat> *g2,
        std::vector<int> *i1, std::vector<int> *i2){
    for(int i = 0; i < data_idx.size(); i++){
        if ((this->train_data->at(data_idx[i]))[col] >= value){
            i1->push_back(data_idx[i]);
            //s1->push_back(data[i]);
            l1->push_back(labels[i]);
            //g1->push_back(seg[i]);
        }
        else{
            i2->push_back(data_idx[i]);
            //s2->push_back(data[i]);
            l2->push_back(labels[i]);
            //g2->push_back(seg[i]);
        }
    }
}

void DecisionTree::finalDivide(
        const std::vector<int> &data_idx, 
        //const std::vector<OutputData> &labels,
        //const std::vector<cv::Mat> &seg,
        int col, InputValue value, 
        std::vector<OutputData> *l1, std::vector<OutputData> *l2,
        //std::vector<cv::Mat> *g1, std::vector<cv::Mat> *g2,
        std::vector<int> *i1, std::vector<int> *i2){
    for(int i = 0; i < data_idx.size(); i++){
        if ((this->train_data->at(data_idx[i]))[col] >= value){
            i1->push_back(data_idx[i]);
            //s1->push_back(data[i]);
            l1->push_back(i);
            //g1->push_back(seg[i]);
        }
        else{
            i2->push_back(data_idx[i]);
            //s2->push_back(data[i]);
            l2->push_back(i);
            //g2->push_back(seg[i]);
        }
    }
}


#define gini3(p) p*p
typedef unsigned int uint32;
// perform actual computation
void forestFindThr( int H, int N, int F, 
  //const float *data,
  const std::vector<std::vector<float>> *data,
  const std::vector<int>&data_idx,
  //const uint32 *hs, 
  std::vector<int> &hs, 
  std::vector<int> &f_idxs,
//const float *ws, 
  //const uint32 *order, 
  int **order,
  //const int split,
  double ws_const,
  int &fid, float &thr, double &gain )
{
  double *Wl, *Wr, *W; //float *data1; 
  int *order1;
  int i, j, j1, j2, h; double vBst, vInit, v, w, wl, wr, g, gl, gr;
  Wl=new double[H]; Wr=new double[H]; W=new double[H];
  // perform initialization
  vBst = vInit = 0; g = 0; w = 0; fid = 1; thr = 0;
  for( i=0; i<H; i++ ) W[i] = 0;

  for( j=0; j<N; j++ ) { 
      w+=ws_const; W[hs[j]]+=ws_const; 
  }
  //for( j=0; j<N; j++ ) { w+=ws[j]; W[hs[j]-1]+=ws[j]; }

  for( i=0; i<H; i++ ) g+=gini3(W[i]); 
  vBst=vInit=(1-g/w/w);
  //printf("-%f %f ini %f bst %f\n", w, g, vInit, vBst);
    //printf("main loop\n");
  // loop over features, then thresholds (data is sorted by feature value)
  for( i=0; i<F; i++ ) {
    //order1=(uint32*) order+i*N; 
      //printf("order1 %d %d\n", i, F);
    order1=order[i]; 
    //data1=(float*) data+i*size_t(N);
    for( j=0; j<H; j++ ) { Wl[j]=0; Wr[j]=W[j]; } 
    gl=wl=0; gr=g; wr=w;

    for( j=0; j<N-1; j++ ) {
      j1=order1[j]; j2=order1[j+1]; 
      //printf("hs %d %d %d\n", j1, N, j);
      h=hs[j1];
      //if(split==0) {
        // gini = 1-\sum_h p_h^2; v = gini_l*pl + gini_r*pr
        wl+=ws_const;//ws[j1]; 
          //printf("Wl %d %d\n", h, H);
        gl-=gini3(Wl[h]); 
        Wl[h]+=ws_const;//ws[j1]; 
        gl+=gini3(Wl[h]);

        wr-=ws_const;//ws[j1]; 
        gr-=gini3(Wr[h]); 
        Wr[h]-=ws_const;//ws[j1]; 
        gr+=gini3(Wr[h]);
        v = (wl-gl/wl)/w + (wr-gr/wr)/w;
      /*} else if (split==1) {
        // entropy = -\sum_h p_h log(p_h); v = entropy_l*pl + entropy_r*pr
        v = gl/w + gr/w;
      } else if (split==2) {
        // twoing: v = pl*pr*\sum_h(|p_h_left - p_h_right|)^2 [slow if H>>0]
        j1=order1[j]; j2=order1[j+1]; h=hs[j1]-1;
        wl+=ws[j1]; Wl[h]+=ws[j1]; wr-=ws[j1]; Wr[h]-=ws[j1];
        g=0; for( int h1=0; h1<H; h1++ ) g+=fabs(Wl[h1]/wl-Wr[h1]/wr);
        v = - wl/w*wr/w*g*g;
      }*/
      //printf("data %d %d %d %d %d\n", j1, j1, data_idx.size(), i, F);
      float d1 = data->at(data_idx[j1])[f_idxs[i]];
      float d2 = data->at(data_idx[j2])[f_idxs[i]];
      if (d2 < d1) {printf("ERRR");}
      //printf("d1 %f\n", d1);
      if( v<vBst && d2 - d1>=1e-6f ) {
      //if( v<vBst && data->at(i)[j2]-data->at(i)[j1]>=1e-6f ) {
        vBst=v; fid=i; thr=0.5f*(d1 + d2); 
      }
        //vBst=v; fid=i+1; thr=0.5f*(data->at(i)[j1]+data->at(i)[j2]); }
        //vBst=v; fid=i+1; thr=0.5f*(data1[j1]+data1[j2]); }
    }
    //printf("n\n");
  }

    //printf("main loop End\n");
  //printf("freee1\n");
  delete [] Wl; 
  //printf("freee2\n");
  delete [] Wr; 
  //printf("freee3\n");
  delete [] W; 
  gain = vInit-vBst;
  //printf("ini %f bst %f\n", vInit, vBst);
  //printf("freee end\n");
}



TreeNode *DecisionTree::buildnode(
        //const std::vector<InputData> &data, 
        const std::vector<int> &data_idx, 
        const std::vector<cv::Mat> &segments, int depth){
    /*for(int i = 0; i < segments.size(); i++){
        char name[100];
        sprintf(name, "seg %d", i % 10);
        printf("%d\n", i);
        cv::Mat tmp;
        cv::normalize(segments[i], tmp, 0, 255, cv::NORM_MINMAX);
        cv::pyrUp(tmp, tmp);
        cv::pyrUp(tmp, tmp);
        cv::imshow(name, tmp);
        if(i%10 == 0) cv::waitKey();
    }*/

    //printf("Start OF node\n");
    int num_of_classes, seg_idx;
    std::vector<int> labels(segments.size(), 0);
#ifdef DECISION_TREE_DEBUG
    printf("Start OF sel %d %d\n", segments.size(), data_idx.size());
#endif
    selectFeaturesFromPatches(segments, &labels, &num_of_classes, &seg_idx);
    //printf("end OF sel\n");

    //double current_score = this->ginii(labels, num_of_classes);
#ifdef DECISION_TREE_DEBUG
    printf("score %f %d depth %d\n", current_score, labels.size(), depth);
#endif
    double best_gain = 0.0;
    //std::vector<InputData>  ms1, ms2;
    //std::vector<cv::Mat> ml1, ml2;
    InputValue best_value;
    int best_col = -1;

    //for(int col = 0; col < this->train_data[0].size(); col++){
    //int m_small = (int)sqrt(this->train_data[0].size());
    int m_small = (this->train_data->at(0).size())/2;

    //int *class_freqsL = new int[num_of_classes];
    //int *class_freqsR = new int[num_of_classes];
    //int num_of_samplesL = 0, num_of_samplesR = 0;
    //int **idxs = getOrderedIdxs(&data, f_idxs, idxs_old);
    //printf("before sort\n");
#ifdef DECISION_TREE_DEBUG
    printf("before sort\n");
#endif
    std::vector<int> f_idxs(m_small); 
    int features_size = this->train_data->at(0).size();
    std::vector<int> fi_vis(features_size, 0);
#ifdef DECISION_TREE_DEBUG
    printf("Choosing features %d\n", m_small);
#endif
    for(int col_idx = 0; col_idx < m_small; ){
        int col = (int)rand() % features_size;
        if(fi_vis[col] != 0){
            continue;
        }
        fi_vis[col]  = 1;
        f_idxs.at(col_idx) = col;
        col_idx++;
    }
    //std::vector<int> idxs_old(data_idx.size());
    //for(int i = 0; i < idxs_old.size(); i++) {idxs_old[i] = i;}
#ifdef DECISION_TREE_DEBUG
    printf("start sort\n");
#endif
    //printf("start sort\n");
    int **idxs = dimSort(this->train_data, f_idxs, &data_idx);
    //int **idxs = dimSort(this->train_data, f_idxs, &data_idx, idxs_old);
    /*for(int f = 0; f < f_idxs.size(); f++){
        int val0 = this->train_data->at(data_idx[idxs[f][0]])[f_idxs[f]];
        for(int i = 1; i < data_idx.size(); i++){
            int id = data_idx[idxs[f][i]];
            int val = this->train_data->at(id)[f_idxs[f]];
            if (val0 > val) printf("ERRRRR\n");
            val0 = val;
        }
    }*/
    //printf("end sort\n");
#ifdef DECISION_TREE_DEBUG
    printf("end sort\n");
#endif

    int fid = -1;
    //printf("Start FFT\n");
    forestFindThr(num_of_classes, 
            data_idx.size(), 
            m_small, 
            this->train_data, 
            data_idx, 
            labels, 
            f_idxs,
            idxs, 
            1.0,///this->train_data->size(),  
            fid, best_value, best_gain);
    printf("node %d %d %f \n", segments.size(), depth, best_gain);
    best_col = f_idxs[fid];
    //printf("ENd FFT\n");
    for(int i = 0; i < f_idxs.size(); i++){
        delete[] idxs[i];
    }
    delete[] idxs;
    //printf("END OF div\n");
    //if(best_gain <= 0)
    //{printf("bad gain %f\n", best_gain);}
    const int min_child = 8;
    if (best_gain > 0 && depth < 64){
        TreeBranch *res = new TreeBranch();
        std::vector<cv::Mat> g1, g2;
        std::vector<OutputData> l1, l2; 
        std::vector<int> ml1, ml2;
        //ml1.clear(); ml2.clear();
        //this->divideSet(data_idx, labels, 
        finalDivide(data_idx, 
                //labels, 
                //segments, 
                best_col, best_value, 
                &l1, &l2, //&g1, &g2, 
                &ml1, &ml2);
        if(ml1.size() > min_child && ml2.size() > min_child){
            //std::vector<InputData> s1, s2;
#ifdef NODE_SHOW_DEBUG
            char name[100];
#endif
            cv::destroyAllWindows();
            for(int i = 0; i < l1.size(); i++){
                g1.push_back(segments[l1[i]]);
#ifdef NODE_SHOW_DEBUG
                if(i < 12){
                    sprintf(name, "g1 %i", i);
                    cv::Mat tmp;
                    cv::pyrUp(g1[i], tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
                    printf("g1 %d %d %d l(%d) %f %f %d\n", i, ml1[i], 
                            best_col, best_value, l1[i],
                            this->train_data->at(ml1[i])[best_col], 
                            this->train_data->at(ml1[i])[best_col] >= best_value);
                    cv::imshow(name, tmp);
                }
#endif                
                //s1.push_back(data[ml1[i]]);
            }
            for(int i = 0; i < l2.size(); i++){
                g2.push_back(segments[l2[i]]);
#ifdef NODE_SHOW_DEBUG
                if(i < 12){
                    sprintf(name, "g2 %i", i);
                    cv::Mat tmp;
                    cv::pyrUp(g2[i], tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
                    printf("g2 %d %d %d l(%d) %f %f %d\n", i, ml2[i], 
                            best_col, best_value, labels[l2[i]],
                            this->train_data->at(ml2[i])[best_col], 
                            this->train_data->at(ml2[i])[best_col] < best_value);
                    cv::imshow(name, tmp);
                }
#endif
                //s2.push_back(data[ml2[i]]);
            }
#ifdef NODE_SHOW_DEBUG
            cv::waitKey();
#endif
            //printf("END OF branch\n");
#ifndef ENABLE_TBB_NODES
            res->left  = buildnode(ml2, g2, depth + 1);
            res->right = buildnode(ml1, g1, depth + 1);
#endif

#ifdef ENABLE_TBB_NODES
            tbb::task_group g;
            if(ml2.size() > 100){
                g.run([&]{
                    res->left  = buildnode(ml2, g2, depth + 1);
                }); // spawn a task
            }else{
                res->left  = buildnode(ml2, g2, depth + 1);
            }
            if(ml1.size() > 100){
                g.run([&]{
                    res->right = buildnode(ml1, g1, depth + 1);
                }); // spawn another task
            }else{
                res->right = buildnode(ml1, g1, depth + 1);
            }
            g.wait();                // wait for both tasks to complete

#endif

            //res->right = buildnode(ms1, ml1);
            res->col = best_col;
            res->value = best_value;
            return res;
        }
        //printf("bad thr\n");
    }
    TreeLeaf *res = new TreeLeaf();
    res->freqs = this->getFreq(labels, num_of_classes);
    res->len = num_of_classes;
    res->patch = segments[seg_idx].clone();
        //printf("END OF leaf\n");
    return res;
}

double DecisionTree::ginii(const std::vector<OutputData> &labels, int num_of_classes){
    int* freqs = this->getFreq(labels, num_of_classes);
    double imp = 0;
    for(int i = 0; i < num_of_classes; i++){
        double p1 = freqs[i]/(double)labels.size();
        for(int j = 0; j < num_of_classes; j++){
            if (i == j) continue;
            double p2 = freqs[j]/(double)labels.size();
            imp += p1*p2;
        }
    }
    delete [] freqs;
    return imp;
}

int DecisionTree::getNumOfClasses(std::vector<OutputData> labels){
    OutputData max = labels[0];
    for(int i = 1; i < labels.size(); i++){
        if (labels[i] > max) max = labels[i];
    }
    return max + 1;
}

int * DecisionTree::getFreq(std::vector<OutputData> labels, int num_of_classes){
    int *res = new int[num_of_classes];
    memset(res, 0, sizeof(OutputData) * num_of_classes);
    for(int i = 0; i < labels.size(); i++){
        res[labels[i]]++;
    }
    return res;
}


DecisionTree::DecisionTree(){
}

TreeNode* readNodeFromFile(std::ifstream &in){
    int type;
    in >> type;
    if(type == 1){
        std::string s;
        in >> s;
        int col ; float value;
        in >> col >> value;
        TreeBranch *b = new TreeBranch();
        b->col = col;
        b->value = value;
        b->left = readNodeFromFile(in);
        b->right = readNodeFromFile(in);
        return b;
    }
    else{
        std::string s;
        in >> s;
        int rows, cols;
        in >> rows >> cols;
        cv::Mat p(rows, cols, CV_8U);
        for(int i = 0; i < p.rows; i++){
            for(int j = 0; j < p.rows; j++){
                int t;
                in >> t;
                p.at<uchar>(i, j) = t;
            }
        }
        TreeLeaf *res = new TreeLeaf();
        res->len = 0;
        res->patch = p;
        return res;
    }

}

void DecisionTree::load(const char *fname){
    std::ifstream in(fname);
    this->head = readNodeFromFile(in);
    in.close();
}

void writeNodeToFile(std::ofstream &out, TreeNode* node){
    if(node->type == 1){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        out << "1\n" << "TreeBranch\n" << b->col << " " << b->value << "\n";
        writeNodeToFile(out, b->left);
        writeNodeToFile(out, b->right);
    }
    else{
        auto p = static_cast<TreeLeaf*>(node)->patch;
        out << "2\n" << "TreeLeaf\n";
        out << p.rows << " " << p.cols << "\n";
        for(int i = 0; i < p.rows; i++){
            for(int j = 0; j < p.rows; j++){
                out << (int)p.at<uchar>(i, j) << " ";
            }
            out << "\n";
        }
    }
}

void DecisionTree::save(const char *fname){
    std::ofstream out(fname);
    writeNodeToFile(out, this->head);
    out.close();
}


/*
    while(node->type != 2){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        if(data[b->col] >= b->value)
            node = b->right;
        else
            node = b->left;
    }
    return static_cast<TreeLeaf*>(node)->patch;
*/
