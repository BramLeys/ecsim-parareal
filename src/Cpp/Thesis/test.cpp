//#include "tests/basic_test.h"
//#include "tests/parareal_convergence_test.h"
//#include "tests/ECSIM_convergence1D_test.h"
//#include "tests/convergence_test.h"
//#include "tests/ECSIM_simulation.h"
//#include "tests/ECSIM_parareal_test.h"
//#include "tests/ECSIM_test1D.h"
//#include "tests/time_step_parameter_test.h"
//#include "tests/time_frame_parameter_test.h"
//#include "tests/subcycling_parameter_test.h"
#include "tests/ECSIM_test3D.h"
//#include "tests/coarse_time_step_parameter_test.h"
//#include "tests/ECSIM3D_simulation.h"

int main(int argc, char* argv[])
{
	return ECSIM_3D_test(argc,argv);
}