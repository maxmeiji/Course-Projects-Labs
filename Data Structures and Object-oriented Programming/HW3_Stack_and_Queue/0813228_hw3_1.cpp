// problem - 1 : I Can Guess the Data Structure (UVa 11995)

/*

	There is a bag-like data structure, supporting two operations:

		1 x Throw an element x into the bag.
		2 Take out an element from the bag.

	Given a sequence of operations with return values, you’re going to guess the data structure. It is
	a stack (Last-In, First-Out), a queue (First-In, First-Out), a priority-queue (Always take out larger
	elements first) or something else that you can hardly imagine!

*/

#include <iostream>
#include <queue>
#include <stack>
using namespace std;

int main()
{

	// input
	int num;
	while (cin >> num)
	{
		queue<int> q;
		priority_queue<int> pq;
		stack<int> s;

		bool isq = true, ispq = true, iss = true;
		while (num--)
		{
			int op, n;
			cin >> op >> n;

			// in
			if (op == 1)
			{
				q.push(n);
				pq.push(n);
				s.push(n);
			}
			else
			{

				if (!q.empty() && q.front() == n)
					q.pop();
				else
					isq = false;

				if (!pq.empty() && pq.top() == n)
					pq.pop();
				else
					ispq = false;

				if (!q.empty() && s.top() == n)
					s.pop();
				else
					iss = false;
			}
		}

		if ((iss && ispq) || (iss && isq) || (isq && ispq))
			cout << "not sure" << endl;
		else if (iss)
			cout << "stack" << endl;
		else if (isq)
			cout << "queue" << endl;
		else if (ispq)
			cout << "priority queue" << endl;
		else
			cout << "impossible" << endl;
	}
}
