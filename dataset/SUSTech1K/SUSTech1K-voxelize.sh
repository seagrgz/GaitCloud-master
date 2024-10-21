#!/bin/bash

split_list="00-nm 01-cr 01-bg 01-ub 01-uf 01-nt 01-nm 01-oc 01-cl"
view_list="000 045 090 135 180 225 270 315 000-far 090-near 180-far 270-far"

for split in $split_list
do
	mkdir /home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel/$split
	for viewpoint in $view_list
	do
		mkdir /home/sx-zhang/SUSTech1K/SUSTech1K-Released-voxel/$split/$viewpoint
		python /home/sx-zhang/work/CNN-LSTM-master/util/gait_voxelize.py -s $split -v $viewpoint
		echo "$split $viewpoint done"
	done
done
