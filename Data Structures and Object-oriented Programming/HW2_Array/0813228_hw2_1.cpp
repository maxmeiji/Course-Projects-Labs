// problem - 1 : Non-decreasing Array (LeetCode [665])

/*
	Given an array nums with n integers, your task is to check if it could become non-decreasing by modifying at most one element.

	We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for every i (0-based) such that (0 <= i <= n - 2).

	Example.
		Input: nums = [4,2,3]
		Output: true
		Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

*/

#include <iostream>
using namespace std;

int main()
{

	int num;
	while (cin >> num)
	{
		cin.ignore();
		int arr[num];
		for (int i = 0; i < num; i++)
		{
			cin >> arr[i];
		}
		cin.ignore();
		int check = 0;
		for (int i = 1; i < num; i++)
		{
			if (check >= 2)
				break;
			if (arr[i] < arr[i - 1])
			{
				check++;
				if (i == 1 || arr[i] > arr[i - 2])
					arr[i - 1] = arr[i];
				else
					arr[i] = arr[i - 1];
			}
		}

		if (check > 1)
		{
			cout << "false\n";
		}
		else
		{
			cout << "true\n";
		}
	}
}
