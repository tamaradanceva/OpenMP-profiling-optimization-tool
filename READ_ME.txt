Note: sample code was also used from https://people.sc.fsu.edu/~jburkardt/cpp_src/dijkstra_openmp/dijkstra_openmp.html
licensed with under the GNU LGPL license.

1) In order to run the application , install OpenCV and SIGAR

  - OpenCV useful links (!There is a bug with FFmpeg files):

	http://opencv.org/ 
	http://www.javieriparraguirre.net/installing-opencv-debian/  

        cmake command if you still have problems with the FFmpeg files, dont make them, you dont need
        those modules for this application to run

        cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_highgui=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_ts=OFF
        http://stackoverflow.com/questions/28699534/exclude-modules-while-building-opencv

  - SIGAR
	https://support.hyperic.com/display/SIGAR/Home#Home-download
2)
// application , the same goes for openmp_algorithm.cpp as omp_application.cpp
g++ omp_application.cpp -fopenmp -o omp_application

3)
// shared library 
g++ -ggdb `pkg-config --cflags --libs opencv` openMP_profiler.cpp -fopenmp
-lsigar -pthread -o openMP_profiler.so -fPIC -shared -ldl -D_GNU_SOURCE

4)
//preload library, replace with correct path 
export LD_PRELOAD="/home/debian/Documents/openMP_profier.so"

5)
// run app
 ./omp_application