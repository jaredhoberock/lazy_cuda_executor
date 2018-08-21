all: demo

demo: demo.cu
	nvcc -O3 -std=c++14 --expt-extended-lambda demo.cu -o demo.out

clean:
	rm -f *.o *.out

