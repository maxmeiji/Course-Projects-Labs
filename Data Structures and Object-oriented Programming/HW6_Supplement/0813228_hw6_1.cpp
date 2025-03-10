#include<iostream>
#include<vector>
#include<unordered_map>
#include<algorithm>
using namespace std;

int main(){
	string s;
	cin >> s;
	if(s.size() < 10){
		cout << "none" << endl;
		return 0;
	}
	unordered_map<string,int> rec;
	for(int i = 0; i <= s.size()-10; i++){
		
		string temp = s.substr(i,10);
		rec[temp]++; 
	}
	vector<string> ans;
	for(auto it : rec){
		if(it.second > 1){
			ans.push_back(it.first);
		}
	}
	if(ans.size() == 0){
		cout << "none" << endl; 
	}
	else{
		sort(ans.begin(),ans.end());
		for(int i = 0; i < ans.size(); i++){
			cout << ans[i] << endl;
		}	
	}

} 
