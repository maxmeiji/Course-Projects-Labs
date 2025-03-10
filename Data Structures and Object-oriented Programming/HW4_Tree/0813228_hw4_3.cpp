#include <iostream>
#include <queue>
#include <unordered_map>
using namespace std;

unordered_map<char, int> map;

struct Min_heap{
    char val;
    int freq;
    Min_heap *left, *right;
    Min_heap(char c, int f) : val(c), freq(f), left(NULL), right(NULL){};
};

class cmp {
    public: 
        bool operator()(Min_heap* a, Min_heap* b) {
   			return a->freq > b->freq;
        }
};

void cal_freq(Min_heap* root, int count){
    if(!root) return;
    if(root->val != '#'){
        map[root->val] *= count;
    }
    else{
        count += 1;
        cal_freq(root->left, count);
        cal_freq(root->right, count);
    }
}
int main(){
    
    // input
    string s;
    getline(cin, s);
    for(int i = 0; i < s.size() - 1; i++){
        map[s[i]]++;
    }

    // use priority queue to always get the first two min_freq elements
    priority_queue<Min_heap* ,vector<Min_heap* >, cmp> pq;
    for(auto n : map){
        pq.push(new Min_heap(n.first, n.second));
    }
    while(pq.size()!=1){
        Min_heap *left, *right, *internal;
        left = pq.top();
        pq.pop();
        right = pq.top();
        pq.pop();

        internal = new Min_heap('#', left -> freq + right -> freq);

        internal -> left = left;
        internal -> right = right;
        pq.push(internal); 
    }

    // result
    cal_freq(pq.top(),0);
    int before = 0, after = 0;
    before = (s.size()-1)*8;
    for(auto n : map){
        after += n.second;
    }

    cout << after << " / "<< before << endl;
}