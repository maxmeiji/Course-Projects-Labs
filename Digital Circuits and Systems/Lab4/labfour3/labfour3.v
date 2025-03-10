module labfour3(SW,LEDR);
input [1:0]SW;
output [0:0]LEDR;

wire R_g,S_g,Qa,Qb,D,Qaa,Qbb,R_gg,S_gg;

nand(R_g,SW[0],~SW[1]);
nand(S_g,~SW[0],~SW[1]);
nand(Qa,R_g,Qb);
nand(Qb,S_g,Qa);

assign D=Qa;
nand(R_gg,D,SW[1]);
nand(S_gg,~D,SW[1]);
nand(Qaa,R_gg,Qbb);
nand(Qbb,S_gg,Qaa);

assign LEDR[0]=Qaa;

endmodule
