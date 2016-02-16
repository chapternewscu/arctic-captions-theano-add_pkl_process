#!/bin/bash
for i in *.jpg

do
	/usr/bin/gm mogrify -resize 256x256^ "$i"
	convert "$i" -resize 256x256^ -gravity Center -crop 224x224+0+0 "$i"
	echo "$i"
done
