#include <iostream>
#include <string.h>
using namespace std;

// problem - 3 : Mutant Flatworld Explorers

/*
	Robotics, robot motion planning, and machine learning are areas that cross the boundaries of many of the subdisciplines that comprise Computer Science: artificial intelligence, algorithms and complexity, electrical and mechanical engineering to name a few. In addition, robots as "turtles" (inspired by work by Papert, Abelson, and diSessa) and as "beeper-pickers" (inspired by work by Pattis) have been studied and used by students as an introduction to programming for many years.
	This problem involves determining the position of a robot exploring a pre-Columbian flat world.
	Given the dimensions of a rectangular grid and a sequence of robot positions and instructions, you are to write a program that determines for each sequence of robot positions and instructions the final position of the robot.

	A robot position consists of a grid coordinate (a pair of integers: 2-coordinate followed by y-coordinate) and an orientation (N,S,E, W for north, south, east, and west). A robot instruction is a string of the letters 'L', 'R', and 'F' which represent, respectively, the instructions:
		• Left: the robot turns left 90 degrees and remains on the current grid point.
		• Right. the robot turns right 90 degrees and remains on the current grid point.
		•Forward: the robot moves forward one grid point in the direction of the current orientation and mantains the same orientation.

	The direction North corresponds to the direction from grid point (∞, y) to grid point (2, y + 1).
	Since the grid is rectangular and bounded, a robot that moves "off" an edge of the grid is lost forever. However, lost robots leave a robot "scent" that prohibits future robots from dropping off the world at the same grid point. The scent is left at the last grid position the robot occupied before disappearing over the edge. An instruction to move "off" the world from a grid point from which a robot has been previously lost is simply ignored by the current robot.

*/

char pos[4] = {'N', 'E', 'S', 'W'};
int main()
{

	int sizex, sizey;
	cin >> sizex >> sizey;
	int map[sizex + 1][sizey + 1];
	memset(map, -1, sizeof map);
	int x, y;
	char ori;

	while (cin >> x >> y >> ori)
	{

		string s;
		cin >> s;
		// cout<<s.size()<<endl;

		int state;
		for (int i = 0; i < 4; i++)
		{
			if (ori == pos[i])
			{
				state = i;
			}
		}
		int over = 0;
		for (int i = 0; i < s.size(); i++)
		{

			switch (s[i])
			{
			case 'R':
				state = (state + 1) % 4;
				break;
			case 'L':
				state = (state - 1);
				if (state < 0)
					state += 4;
				break;

			case 'F':
				if (state == map[x][y])
				{
					break;
				}
				if (state == 0)
				{
					if (y + 1 > sizey)
					{
						over = 1;
						break;
					}
					y += 1;
				}
				else if (state == 1)
				{
					if (x + 1 > sizex)
					{
						over = 1;
						break;
					}
					x += 1;
				}
				else if (state == 2)
				{
					if (y - 1 < 0)
					{
						over = 1;
						break;
					}
					y -= 1;
				}
				else
				{
					if (x - 1 < 0)
					{
						over = 1;
						break;
					}
					x -= 1;
				}
				break;
			}
			if (over == 1)
			{
				map[x][y] = state;
				break;
			}

			// cout<<x<<" "<<y<<" "<<state<<endl;
		}
		cout << x << " " << y << " " << pos[state];
		if (over == 1)
		{
			cout << " LOST" << endl;
		}
		else
		{
			cout << endl;
		}
	}
}
