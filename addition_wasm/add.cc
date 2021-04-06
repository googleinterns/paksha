#include <iostream>
#include "add.h"

extern "C" {
  int addNums(int a, int b) {
    Add addition;

    // Set up args and run the computation.
    const float args[2] = {(float)a, (float)b};
    std::copy(args + 0, args + 1, addition.arg0_data());
    std::copy(args + 1, args + 2, addition.arg1_data());
    addition.Run();

    return addition.result0();
  }

}
