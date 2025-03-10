#include<iostream>
#include<vector>
using namespace std;

int main(){
	vector<vector<char> > grid;
	for(int i = 0; i < 9; i++){
		vector<char> input;
		for(int j = 0; j < 9; j++){
			char temp;
			cin >> temp;
			input.push_back(temp);
		}
		grid.push_back(input);
	}
	
	int row[9][9] = {0}, col[9][9] = {0}, box[9][9] = {0};
	
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			if(grid[i][j] != 'x'){
				int temp = grid[i][j] - '0' - 1;
				if(row[i][temp] == 1 || col[j][temp] == 1 || box[i/3*3+j/3][temp] == 1){
					cout << "false" << endl;
					return 0;
				}
				row[i][temp] = col[j][temp] = box[i/3*3+j/3][temp] = 1 ;
			}
		}
	}
	cout << "true" << endl; 
	return 0;	
}
