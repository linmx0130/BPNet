all: datagen main
datagen: datagen.cpp
	g++ -o datagen datagen.cpp
main: main.cpp linear_algebra.h
	g++ -o main main.cpp -O2
clean : 
	rm datagen main
