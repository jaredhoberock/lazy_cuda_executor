all: demo demo2 demo3

demo: demo.cu
	nvcc -O3 -std=c++14 --expt-extended-lambda demo.cu -o demo.out

demo2: demo2.cu
	nvcc -O3 -std=c++14 --expt-extended-lambda demo2.cu -o demo2.out

demo3: demo3.cu
	nvcc -O3 -std=c++14 --expt-extended-lambda demo3.cu -o demo3.out

clean:
	rm -f *.o *.out

