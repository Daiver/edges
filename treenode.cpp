#include "treenode.h"
#include <stdio.h>

void TreeBranch::show(int n){
    for(int i = 0; i < n; i++)
        printf("  ");
    printf("br %d %d\n", this->col, this->value);
    this->left->show(n+1);
    this->right->show(n+1);
}

void TreeLeaf::show(int n){
    for(int i = 0; i < n; i++)
        printf("  ");
    printf("l [");
    for(int i = 0; i < this->len; i++) printf("%d ", this->freqs[i]);
    printf("]\n");
}
