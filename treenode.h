#ifndef __DTREETYPEDEFS_H__
#define __DTREETYPEDEFS_H__

#include "dtreetypedefs.h"
#include <stdio.h>

class TreeNode{
    public:
        int type;
        virtual void show(int n=0) = 0;
};

class TreeLeaf : public TreeNode{
    public:
        int* freqs;
        int len;
        TreeLeaf(){
            this->type = 2;
        }
        void show(int n=0);
};

class TreeBranch : public TreeNode{
    public:
        TreeBranch() {
            this->type = 1;
        }
        void show(int n=0);
        InputValue value;
        int col;
        TreeNode* left;
        TreeNode* right;
};
#endif
