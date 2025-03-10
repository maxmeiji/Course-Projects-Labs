// problem - 3 : Polynomial Equation

/*
	A Polynomial equation of degree n is defined as follows :

	A polynomial equation of n degree can have at most n distinct roots which may be both real or
	complex. Such as a quadratic equation :

	x^2 − 5x + 6 = 0
	has two roots 2 and 3. In this problem you have to generate such a polynomial equation whose roots
	are already given.

*/

#include <iostream>
#include <cstdlib>
using namespace std;

int main()
{
	int num;
	while (cin >> num)
	{
		int root[num];
		for (int i = 0; i < num; i++)
		{
			cin >> root[i];
		}
		int coef[num + 1] = {0};
		coef[1] = 1;
		coef[0] = -root[0];

		for (int i = 1; i < num; i++)
		{
			int temp[num + 1] = {0};
			int mul[2];
			mul[1] = 1;
			mul[0] = -root[i];
			for (int j = 0; j <= num; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					temp[k + j] += coef[j] * mul[k];
				}
			}
			for (int j = 0; j < num + 1; j++)
			{
				coef[j] = temp[j];
			}
		}

		int sig = 0;
		for (int i = num; i >= 1; i--)
		{
			if (coef[i] != 0)
			{
				if (sig == 0)
				{
					sig = 1;
				}
				else
				{
					if (coef[i] > 0)
						cout << "+ ";
					else
						cout << "- ";
				}
				if (i > 1)
				{

					if (abs(coef[i]) == 1)
					{
						cout << "x^" << i << " ";
					}
					else
					{
						cout << abs(coef[i]) << "x^" << i << " ";
					}
				}
				else
				{
					if (abs(coef[i]) == 1)
					{
						cout << "x ";
					}
					else
					{
						cout << abs(coef[i]) << "x ";
					}
				}
			}
		}
		if (coef[0] >= 0)
			cout << "+ ";
		else
			cout << "- ";
		cout << abs(coef[0]) << " = 0" << endl;
	}
}
