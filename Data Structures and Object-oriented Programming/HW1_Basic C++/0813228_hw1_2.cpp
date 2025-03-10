#include <iostream>
#include <cmath>
using namespace std;

// problem - 2 : Symmetric Matrix

/*

You're given a square matrix M. Elements of this matrix are Mij: {0 < i < n, 0 < j < n}. 
In this problem you'll have to find out whether the given matrix is symmetric or not.
Definition: Symmetric matrix is such a matrix that all elements of it are non- negative and symmetric with relation to the center of this matrix. 
Any other matrix is considered to be non-symmetric.

*/

int main()
{
	int num;
	cin >> num;
	int T = num;
	while (num--)
	{
		char nan_a, nan_b;
		int N;
		cin >> nan_a >> nan_b >> N;

		long long matrix[N][N];
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				cin >> matrix[i][j];
			}
		}

		float mid = float(N - 1) / 2;
		int ans = 0;

		for (int i = 0; i <= ceil(double(N) * N / 2) - 1; i++)
		{
			int x = i / N;
			int y = i % N;
			int compx = mid + (mid - x);
			int compy = mid + (mid - y);
			// cout<<matrix[x][y]<<" "<<compx<<" "<<compy<<endl;
			if (matrix[x][y] != matrix[compx][compy] || matrix[x][y] < 0)
			{
				ans = -1;
				break;
			}
		}

		if (ans == -1)
		{
			cout << "Test #" << (T - num) << ": Non-symmetric." << endl;
		}
		else
		{
			cout << "Test #" << (T - num) << ": Symmetric." << endl;
		}
	}
}
