// RUN: circt-translate %s --emit-hgldd

#loc1 = loc("test/DebugDoodle.fir":2:10)
#loc2 = loc("test/DebugDoodle.fir":3:11)
#loc3 = loc("test/DebugDoodle.fir":4:12)
#loc4 = loc("test/DebugDoodle.fir":6:15)

hw.module @Foo(%a: i32 loc(#loc2)) -> (b: i32 loc(#loc3)) {
  %0 = comb.mul %a, %a : i32 loc(#loc4)
  hw.output %0 : i32 loc(#loc1)
} loc(#loc1)

