TARGET = lstm
OBJECT = main.o cornernn_math.o
CFLAGS = -O2 -std=c++11
CC = g++

all: $(TARGET)

lstm: $(OBJECT)
	$(CC) $(CFLAGS) -o $@ $^

main.o: main.cpp 
	$(CC) $(CFLAGS) -c -o $@ $^

cornernn_math.o: cornernn_math.cpp
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm $(TARGET) $(OBJECT)
