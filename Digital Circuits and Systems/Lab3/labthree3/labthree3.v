module labthree3(SW, LEDR);
input [9:0]SW;
 
wire [3:0]s;
wire [3:0]c;
output [4:0]LEDR;

full_adder f1(SW[0], SW[4], SW[9], c[0], s[0]);
full_adder f2(SW[1], SW[5], c[0], c[1], s[1]);
full_adder f3(SW[2], SW[6], c[1], c[2], s[2]);
full_adder f4(SW[3], SW[7], c[2], c[3], s[3]);

assign LEDR[0]=s[0];
assign LEDR[1]=s[1];
assign LEDR[2]=s[2];
assign LEDR[3]=s[3];
assign LEDR[4]=c[3];

endmodule

module full_adder(a,b,cin,cout,sout);
input a;
input b;
input cin;
output cout;
output sout;
wire C1, C2, S1;

process(a,b,C1,S1);
process(S1,cin,C2,sout);
assign cout=C1|C2;

endmodule

module process(x,y,C,S);
input x,y;
output C, S;

assign C=(x&y);
assign S=(~x&y)|(x&~y);

endmodule