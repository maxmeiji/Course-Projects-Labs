// problem - 2 : Validate Stack Sequences (https://www.geeksforgeeks.org/check-if-the-given-push-and-pop-sequences-of-stack-is-valid-or-not/)

/*
	Given an integer length and two integer sequences pushed and popped each with distinct values,
	print out “true” if this could have been the result of a sequence of push and pop operations on an initially empty stack, or “false” otherwise.
	The input is terminated by end-of-file (EOF).

*/
#include <iostream>
#include <stack>
using namespace std;

int main()
{
	int num;
	while (cin >> num)
	{
		int in[num];
		int out[num];
		for (int i = 0; i < num; i++)
		{
			cin >> in[i];
		}
		for (int j = 0; j < num; j++)
		{
			cin >> out[j];
		}

		int rec = 0;
		stack<int> store;
		for (int i = 0; i < num; i++)
		{
			store.push(in[i]);
			while (!store.empty() && rec < num && store.top() == out[rec])
			{
				store.pop();
				rec++;
			}
		}
		if (rec == num)
			cout << "true" << endl;
		else
			cout << "false" << endl;
	}
}
