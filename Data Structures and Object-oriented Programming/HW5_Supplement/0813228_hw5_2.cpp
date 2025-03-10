#include<iostream>
#include<algorithm>
#include<queue>
using namespace std;

int main(){
	int n, d;
	while(cin >> n >> d){
		vector<int>exp;
		for(int i = 0; i < n; i++){
			int temp;
			cin >> temp;
			exp.push_back(temp);
		}
		int max_val = *max_element(exp.begin(), exp.end());
		int count[max_val] = {0};
		for(int i = 0; i < d; i++){
			count[exp[i]]++;
		}
		int rec = 0;
		int fin = 0;
		int med = 0;
		int ans = 0;
		for(int i = d; i < n; i++){

			fin = 0;
			rec = 0;
			for(int j = 0; j < max_val; j++){
				if(fin == 1) break;
				rec += count[j];
				if(d%2==0){
					if(rec >= d/2){
						if(rec > d/2 && fin == 0){
							med = j*2;
							fin = 1;
						}
						else if(rec == d/2){
							med = j;
							fin = 2;
						}
						else{
							med += j;
							fin = 1;
						}
						
					}
				}
				else{
					if(rec > d/2){
						med = j*2;
						fin = 1;
					}
				}
			}
			if(exp[i] >= med) ans++;
			count[exp[i-d]]--;
			count[exp[i]]++;
		}
		cout << ans << endl;
		break;
	}
}
