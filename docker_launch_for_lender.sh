#!/bin/bash
docker run -v $PWD/logs:/cars/logs -v $PWD/save_animation_folder:/cars/save_animation_folder -it $1 python entrypoint_lunar.py --name $2
