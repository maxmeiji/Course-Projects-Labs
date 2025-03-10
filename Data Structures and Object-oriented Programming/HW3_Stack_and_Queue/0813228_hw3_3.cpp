// problem - 3 : Bicolorable (UNa 10004)

/*
	In 1976 the “Four Color Map Theorem” was proven with the assistance of a computer.
	This theorem states that every map can be colored using only four colors, in such a way that no region is colored using the same color as a neighbor region.


	Here you are asked to solve a simpler similar problem. You have to decide whether a given arbitrary connected graph can be bicolored.
	That is, if one can assign colors (from a palette of two) to the nodes in such a way that no two adjacent nodes have the same color.
	To simplify the problem you can assume:
		no node will have an edge to itself.
		the graph is nondirected. That is, if a node a is said to be connected to a node b, then you must assume that b is connected to a.
		the graph will be strongly connected. That is, there will be at least one path from any node to any other node.

*/

#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int main()
{
	int n;
	while (cin >> n, n != 0)
	{
		// initialize colors
		int colors[n];
		for (int i = 0; i < n; i++)
		{
			colors[i] = -1;
		}

		// initialize line
		int line, a, b;
		vector<int> rec[n];
		cin >> line;
		for (int i = 0; i < line; i++)
		{

			cin >> a >> b;
			rec[a].push_back(b);
			rec[b].push_back(a);
		}

		queue<int> task;
		task.push(a);
		colors[a] = 1;
		int ans = 1;
		while (!task.empty())
		{
			int temp = task.front();
			task.pop();
			for (int i = 0; i < rec[temp].size(); i++)
			{
				int child = rec[temp][i];
				if (colors[child] == colors[temp])
				{
					ans = -1;
					break;
				}
				if (colors[child] == -1)
				{
					colors[child] = 1 - colors[temp];
					task.push(child);
				}
			}
		}
		if (ans == -1)
			cout << "NOT BICOLORABLE." << endl;
		else
			cout << "BICOLORABLE." << endl;
	}
}
