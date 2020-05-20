#!/bin/bash 
train="train"
test_="test"


for i in {0..102..1};
do 
    init=$i
    fin=$[$i + 1]
    python3 ResNet-50_tf_keras-v6tmp.py $init $fin $train
done

for i in {0..33..1};
do 
    init=$i
    fin=$[$i + 1]
    python3 ResNet-50_tf_keras-v6tmp.py $init $fin $test_
done
