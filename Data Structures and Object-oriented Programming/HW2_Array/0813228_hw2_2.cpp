// problem - 2 : Solve the Equation (LeetCode [640])

/*
	Solve a given equation and return the value of 'x' in the form of a string "x=#value". The equation contains only '+', '-' operation, the variable 'x' and its coefficient. You should return "No solution" if there is no solution for the equation, or "Infinite solutions" if there are infinite solutions for the equation.

	If there is exactly one solution for the equation, we ensure that the value of 'x' is an integer.

	Example 1:

		Input: equation = "x+5-3+x=6+x-2"
		Output: "x=2"

*/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

vector<int> x;
vector<int> constant;
vector<int> temp;
int sig = 1;

void process(char c)
{
	if (int(c) >= 48 && int(c) <= 57)
	{
		temp.push_back(int(c) - 48);
	}
	else if (c == 'x')
	{
		if (temp.size() == 0)
		{
			x.push_back(1 * sig);
			sig = 1;
		}
		else
		{
			int tot = 0;
			for (int i = 0; i < temp.size(); i++)
			{
				tot += temp[i] * pow(10, temp.size() - i - 1);
			}
			tot *= sig;
			temp.clear();
			x.push_back(tot);
			sig = 1;
		}
	}
	else
	{

		int tot = 0;
		for (int i = 0; i < temp.size(); i++)
		{
			tot += temp[i] * pow(10, temp.size() - i - 1);
		}
		tot *= sig;
		temp.clear();
		constant.push_back(tot);
		if (c == '+')
			sig = 1;
		else if (c == '-')
			sig = -1;
		else
			sig = 1;
	}
}

int main()
{

	int cases;
	cin >> cases;
	// cin.ignore();
	while (cases--)
	{

		string s;
		cin >> s;
		// cin.ignore();
		// ax+b = cx+d
		int a = 0, b = 0, c = 0, d = 0;
		for (int i = 0; i < s.length(); i++)
		{
			process(s[i]);
			if (s[i] == '=')
			{
				for (int j = 0; j < x.size(); j++)
				{
					a += x[j];
				}
				for (int j = 0; j < constant.size(); j++)
				{
					b += constant[j];
				}
				x.clear();
				constant.clear();
			}
		}
		if (temp.size() > 0)
		{
			int tot = 0;
			for (int i = 0; i < temp.size(); i++)
			{
				tot += temp[i] * pow(10, temp.size() - i - 1);
			}
			tot *= sig;
			temp.clear();
			constant.push_back(tot);
			sig = 1;
		}
		for (int i = 0; i < x.size(); i++)
		{
			c += x[i];
		}
		for (int i = 0; i < constant.size(); i++)
		{
			d += constant[i];
		}
		x.clear();
		constant.clear();
		if ((a - c) != 0)
		{
			int ans;
			ans = floor(double(d - b) / (a - c));
			cout << ans << endl;
		}
		else
		{
			if ((d - b) == 0)
			{
				cout << "IDENTITY" << endl;
			}
			else
			{
				cout << "IMPOSSIBLE" << endl;
			}
		}
	}
}
