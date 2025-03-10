#include<iostream>
#include<queue>
using namespace std;
class cmp {
    public: 
        bool operator()(pair<int,int> a, pair<int,int> b) {
   			return a.second>b.second;
        }
};
int main(){
    int id, st, time;
    vector<pair<int, int> > rec;
    vector<int> index;
    while(cin >> id >> st >> time){
        index.push_back(id);
        rec.push_back({st,time});
    }
    priority_queue<pair<int,int>, vector<pair<int,int> >, cmp> pq;
    for(int i = 0; ; i++){
        int this_one;
        if(!pq.empty()){
            this_one = pq.top().first;
        }
        else{
            this_one = -1;
        }

        if(pq.empty() && i > rec[rec.size()-1].second) break; 
        if(!pq.empty()){
            pair<int, int> temp;
            temp = pq.top();
            temp.second--;
            if(temp.second == 0) pq.pop();
            else{
                pq.pop();
                pq.push(temp);
            }

        }
        for(int k = 0; k < index.size(); k++){
            if(rec[k].first==i){
                pq.push(make_pair(index[k],rec[k].second));
            }

        }


        //cout << pq.top().first << " "<<pq.top().second<<endl;
        if(!pq.empty() && (pq.top().first != this_one || this_one == -1)) cout << pq.top().first<<endl;
    }

}
