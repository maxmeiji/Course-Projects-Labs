#include<iostream>
#include<queue>
#include<sstream>
using namespace std;

struct TreeNode{
	int val;
	int depth;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x):val(x),depth(0),left(NULL),right(NULL){};	
};

int update_height(TreeNode* root){
	if(!root) return -1; 
	else root->depth = max(update_height(root->left), update_height(root->right)) + 1;
	return root->depth;
}

TreeNode* right_rot(TreeNode* root){
	TreeNode* temp = root->left;
	TreeNode* following = temp->right;
	temp->right = root;
	root->left = following;
	return temp;
}

TreeNode* left_rot(TreeNode* root){
	TreeNode* temp = root->right;
	TreeNode* following = temp->left;
	temp->left = root;
	root->right = following;
	return temp;
}

int depth_diff(TreeNode* root){
	// judge whether the tree is unbalanced 
	int right_depth = (root->right) ? root->right->depth : -1;
	int left_depth = (root->left) ? root->left->depth : -1;
	int height_diff = left_depth - right_depth;
	return height_diff;
}

TreeNode* avl_update(TreeNode* root){
	int height_depth = depth_diff(root);
	// rotation
	if(height_depth > 1){
		if(depth_diff(root->left) == 1){
			// LL
			root = right_rot(root);
		}
		else{
			// LR
			root->left = left_rot(root->left);
			root = right_rot(root);				
		}
	}	
	else if(height_depth < -1){
		if(depth_diff(root->right) == -1){
			// RR
			root = left_rot(root);
		}
		else{
			// RL
			root->right = right_rot(root->right);
			root = left_rot(root);
		}
	}
	return root;		
}

TreeNode* insert(TreeNode* root, int x){
	if(!root) return NULL;
	if(x > root->val){
		if(!root->right) root->right = new TreeNode(x);
		else root->right = insert(root->right, x);
	}
	else{
		if(!root->left) root->left = new TreeNode(x);
		else root->left = insert(root->left, x); 
	}
	
	root->depth = update_height(root);
	root = avl_update(root);
	return root;
}




int main(){
	string s;
	getline(cin, s);
	stringstream ss(s);
	int temp;
	ss >> temp;
	TreeNode* root = new TreeNode(temp);
	while(true){
		ss >> temp;
		if(ss.fail())break;
		root = insert(root,temp);
	}
	
	vector<vector<int> > rec;
	queue<TreeNode*> q;
	q.push(root);
	while(!q.empty()){
		int size = q.size();
		vector<int> level_arr;
		for(int i = 0; i < size; i++){
			TreeNode* temp = q.front();
			q.pop();
			level_arr.push_back(temp->val);
			if(temp->left) q.push(temp->left);
			if(temp->right) q.push(temp->right);
		}
		rec.push_back(level_arr);
	}
	for(int i = 0; i < rec.size(); i++){
		for(int j = 0; j < rec[i].size(); j++){
			if(i == rec.size()-1 && j == rec[i].size()-1) cout << rec[i][j] << endl;
			else cout << rec[i][j] << " ";
		}
	}
}
