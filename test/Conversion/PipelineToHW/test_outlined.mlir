// RUN: circt-opt --lower-pipeline-to-hw="outline-stages" %s | FileCheck %s

// CHECK-LABEL:   hw.module @testBasic_p0(
// CHECK-SAME:          %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out0: i1) {
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testBasic(
// CHECK-SAME:          %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out: i1) {
// CHECK:           %[[VAL_3:.*]] = hw.instance "testBasic_p0" @testBasic_p0(in0: %[[VAL_0]]: i1, clk: %[[VAL_1]]: i1, rst: %[[VAL_2]]: i1) -> (out0: i1)
// CHECK:           hw.output %[[VAL_3]] : i1
// CHECK:         }

hw.module @testBasic(%arg0: i1, %clk: i1, %rst: i1) -> (out: i1) {
  %0 = pipeline.scheduled(%arg0) clock %clk reset %rst : (i1) -> (i1) {
  ^bb0(%a0: i1):
    pipeline.return %a0 : i1
  }
  hw.output %0 : i1
}

// CHECK-LABEL:   hw.module @testSingle_p0(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testSingle_p0_s0" @testSingle_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32)
// CHECK:           %[[VAL_7:.*]] = hw.constant true
// CHECK:           %[[VAL_8:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           hw.output %[[VAL_8]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle_p0_s0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           hw.output %[[VAL_7]], %[[VAL_8]] : i32, i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testSingle_p0" @testSingle_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32, i1) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %true = hw.constant true
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32) enable %arg2
  ^bb1(%6: i32, %7: i32):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8, %true : i32, i1
  }
  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testMultiple_p0(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testMultiple_p0_s0" @testMultiple_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, out2: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = hw.instance "testMultiple_p0_s1" @testMultiple_p0_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, in2: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32)
// CHECK:           %[[VAL_10:.*]] = hw.constant true
// CHECK:           %[[VAL_11:.*]] = comb.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_11]], %[[VAL_10]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p0_s0(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, out2: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p0_s1(
// CHECK-SAME:              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]] : i32, i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testMultiple_p1_s0" @testMultiple_p1_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, out2: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = hw.instance "testMultiple_p1_s1" @testMultiple_p1_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, in2: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32)
// CHECK:           %[[VAL_10:.*]] = hw.constant true
// CHECK:           %[[VAL_11:.*]] = comb.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_11]], %[[VAL_10]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1_s0(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, out2: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testMultiple_p0" @testMultiple_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = hw.instance "testMultiple_p1" @testMultiple_p1(in0: %[[VAL_5]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           hw.output %[[VAL_5]], %[[VAL_6]] : i32, i1
// CHECK:         }
hw.module @testMultiple(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32, i1) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %true = hw.constant true
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0, %arg2 : i32, i32, i1) enable %arg2
  ^bb1(%2: i32, %3: i32, %4: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32) enable %4
  ^bb2(%6: i32, %7: i32):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8, %true : i32, i1
  }

  %1:2 = pipeline.scheduled(%0#0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32, i1) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %true = hw.constant true
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0, %arg2 : i32, i32, i1) enable %arg2
  ^bb1(%2: i32, %3: i32, %4: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32) enable %4
  ^bb2(%6: i32, %7: i32):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8, %true : i32, i1
  }

  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testSingleWithExt_p0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.instance "testSingleWithExt_p0_s0" @testSingleWithExt_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32)
// CHECK:           %[[VAL_6:.*]] = hw.instance "testSingleWithExt_p0_s1" @testSingleWithExt_p0_s1(in0: %[[VAL_5]]: i32, extIn0: %[[VAL_2]]: i32, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32)
// CHECK:           hw.output %[[VAL_6]], %[[VAL_2]] : i32, i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt_p0_s0(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_4:.*]] = hw.constant true
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           hw.output %[[VAL_6]] : i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt_p0_s1(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_4:.*]] = hw.constant true
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           hw.output %[[VAL_6]] : i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingleWithExt(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = hw.instance "testSingleWithExt_p0" @testSingleWithExt_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_0]]: i32, extIn0: %[[VAL_1]]: i32, clk: %[[VAL_2]]: i1, rst: %[[VAL_3]]: i1) -> (out0: i32, out1: i32)
// CHECK:           hw.output %[[VAL_4]], %[[VAL_5]] : i32, i32
// CHECK:         }
hw.module @testSingleWithExt(%arg0: i32, %ext1: i32, %clk: i1, %rst: i1) -> (out0: i32, out1: i32) {
  %0:2 = pipeline.scheduled(%arg0, %arg0) ext (%ext1 : i32) clock %clk reset %rst : (i32, i32) -> (i32, i32) {
  ^bb0(%a0: i32, %a1 : i32, %ext0: i32):
    %true = hw.constant true
    %1 = comb.sub %a0, %a0 : i32
    pipeline.stage ^bb1 regs(%1 : i32) enable %true

  ^bb1(%6: i32):
    // Use the external value inside a stage
    %8 = comb.add %6, %ext0 : i32
    pipeline.stage ^bb2 regs(%8 : i32) enable %true
  
  ^bb2(%9 : i32):
  // Use the external value in the exit stage.
    pipeline.return %9, %ext0  : i32, i32
  }
  hw.output %0#0, %0#1 : i32, i32
}
