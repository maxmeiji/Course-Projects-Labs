#include<iostream>
using namespace std;

int main(){
	
	int num;
	while(cin>>num, num!=0){
		int arr[num];
		for(int i = 0; i < num; i++){
			int x;
			cin >> x;
			arr[i] = x;
		}
		int swap = 0;
		for(int i = num-1; i > 0; i--){
			for(int j = 0; j <i; j++){
				if(arr[j] > arr[j+1]){
					swap++;
					int temp = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = temp;
				}
			}
		}
		cout << swap << endl;	
	}
} 
