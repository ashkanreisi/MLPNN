CPP = hipcc

all:source

source: MLPNN.cpp
	$(CPP) $(CFLAGS) -o source MLPNN.cpp

clean:
	rm -f source *.o




