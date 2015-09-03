#/usr/bin/sh
make 
./datagen >input
./main >log
echo "INPUT FILE = input AND LOG FILE = log"
