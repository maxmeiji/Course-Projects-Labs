module labfour1(input Sbar, Rbar, output Q, Qbar);
  nand LS(Q, Sbar, Qbar);
  nand LR(Qbar, Rbar, Q);
endmodule

module enLatch(input en, S, R, output Q, Qbar);
  nand ES(Senbar, en, S);
  nand ER(Renbar, en, R);
  labfour1 L1(Senbar, Renbar, Q, Qbar);
endmodule

module main;
reg S, en, R;
wire Q, Qbar;

enLatch enLatch1(en, S, R, Q, Qbar);
endmodule
