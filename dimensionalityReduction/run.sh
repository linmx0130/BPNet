#!/usr/bin/sh
g++ -o datagen datagen.cpp
g++ -o main main.cpp
./datagen input
if [ $? -ne 0 ] 
then
    echo "Error Input!"
else 
    ./main > result
fi
rm datagen main
