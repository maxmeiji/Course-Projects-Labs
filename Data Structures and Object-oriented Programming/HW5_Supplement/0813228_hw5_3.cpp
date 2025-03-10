#include<iostream>
#include<queue>
#include<vector>
#include<climits>
using namespace std;

struct Edge{
	int from;
	int end;	
	int weight;
};

void BF(int st, int ed, int n, vector<Edge> arr){
	int dis[n];
	for(int i = 0; i < n; i++){
		dis[i] = INT_MAX;
	}
	dis[st] = 0;
	
	// Bellman-Ford 
	for(int i = 0; i < n-1; i++){
		for(int j = 0; j < arr.size(); j++){
			Edge temp = arr[j];
			int from = temp.from;
			int end = temp.end;
			int weight = temp.weight;
			if(dis[from] != INT_MAX && dis[end] > dis[from] + weight){
				dis[end] = dis[from] +weight;
			}
		}
	}
	int ans = dis[ed];
	for(int i = 0; i < arr.size(); i++){
		Edge temp = arr[i];
		int from = temp.from;
		int end = temp.end;
		int weight = temp.weight;
		if(dis[from] != INT_MAX && dis[from] + weight < dis[end]){
			dis[end] = dis[from] +weight;
			ans = -INT_MAX;
		}
	}
	if(ans == -INT_MAX) cout << "INF" << endl;
	else if(ans == INT_MAX) cout << "UNREACHABLE" << endl;
	else cout << ans << endl;
	
}
int main(){
	int num;
	cin >> num;
	while(num--){
		int node_num, edge_num, st, ed;
		cin >> node_num >> edge_num;
		cin >> st >> ed;
		vector<Edge> arr;
		for(int i = 0; i < edge_num; i++){
			int f, s, w;
			cin >> f >> s >> w;
			Edge edge;
			edge.from = f;
			edge.end = s; 
			edge.weight = w;
			arr.push_back(edge);
		}
		BF(st, ed, node_num, arr);
	
	}	
}
