module labfour2(SW,LEDR);

input [1:0]SW;
output [0:0]LEDR;

wire R_g,S_g,Qa,Qb;

nand(R_g,SW[0],SW[1]);
nand(S_g,~SW[0],SW[1]);
nand(Qa,R_g,Qb);
nand(Qb,S_g,Qa);

assign LEDR[0]=Qa;

endmodule