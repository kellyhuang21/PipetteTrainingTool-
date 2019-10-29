#!/bin/bash

cd data/
DATE=${1?Error: Date not supplied in the format: 2019-02-2};
count=0; 
for file in $(ls); 
do if [[ $file == *"$DATE"* ]]; then echo $file; count=$((count + 1)) ; fi; done; echo $count

