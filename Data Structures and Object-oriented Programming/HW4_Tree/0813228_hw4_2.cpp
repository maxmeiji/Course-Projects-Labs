#include <iostream>
#include <vector>
using namespace std;

struct BST
{
    int val;
    BST *left, *right;
    BST() : val(0), left(NULL), right(NULL){};
    BST(int x) : val(x), left(NULL), right(NULL){};
};

BST *insert(BST *root, int x)
{
    if (!root)
        return new BST(x);
    if (root->val > x)
        root->left = insert(root->left, x);
    else
        root->right = insert(root->right, x);
    return root;
}

bool search(BST *root, int x, vector<int> &ans)
{

    if (!root)
        return false;
    ans.push_back(root->val);
    if (root->val == x)
        return true;
    else if (root->val > x)
        return search(root->left, x, ans);
    else
        return search(root->right, x, ans);
}

BST *delete_node(BST *root, int x)
{
    if (!root)
        return NULL;
    if (root->val > x)
        root->left = delete_node(root->left, x);
    else if (root->val < x)
        root->right = delete_node(root->right, x);
    else
    {
        // case 1 : no child
        if (!root->left && !root->right)
            return NULL;
        // case 2 : single-side child
        else if (!root->right)
        {
            BST *temp = root->left;
            while (temp->right)
                temp = temp->right;
            root->val = temp->val;
            root->left = delete_node(root->left, temp->val);
        }
        else
        {
            // case 3 : two-side child
            BST *temp = root->right;
            while (temp->left)
                temp = temp->left;
            root->val = temp->val;
            root->right = delete_node(root->right, temp->val);
        }
    }
    return root;
}

int main()
{
    BST *root = NULL;
    string s;
    int num;
    while (cin >> s >> num)
    {
        cin.ignore();
        if (s == "add")
        {
            root = insert(root, num);
        }
        else if (s == "delete")
        {
            vector<int> ans;
            if (!search(root, num, ans))
            {
                cout << "not found" << endl;
            }
            else
            {
                root = delete_node(root, num);
            }
        }
        else
        {
            bool success;
            vector<int> ans;
            success = search(root, num, ans);
            if (ans.size() >= 1)
            {
                for (int i = 0; i < ans.size() - 1; i++)
                {
                    cout << ans[i] << " ";
                }
                cout << ans[ans.size() - 1];
                if (!success)
                    cout << " not found" << endl;
                else
                    cout << endl;
            }
            else
            {
                cout << "not found" << endl;
            }
        }
    }
}