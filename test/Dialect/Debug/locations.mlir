// RUN: circt-opt %s --verify-diagnostics | circt-opt | FileCheck %s

// CHECK: module

#loc1 = #dbg.magic<"hello", loc(unknown)>
unrealized_conversion_cast to i1 loc(#loc1)
