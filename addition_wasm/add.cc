#include <iostream>
#include "add.h"

int main(int argc, char** argv) {
  Add addition;

  // Set up args and run the computation.
  const float args[2] = {1, 3};
  std::copy(args + 0, args + 1, addition.arg0_data());
  std::copy(args + 1, args + 2, addition.arg1_data());
  addition.Run();

  // Check result
  if (addition.result0() == 4) {
    std::cout << "Success! Found expected value of 4." << std::endl;
  } else {
    std::cout << "Failed. Expected value 4 at 0,0. Got:"
              << addition.result0() << std::endl;
  }

  return 0;
}
