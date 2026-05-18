// testing structural map

#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_map.h>

TEST(StructuralMap, Basic) {
  EXPECT_EQ(StructuralMap::Map(1), 1);
}