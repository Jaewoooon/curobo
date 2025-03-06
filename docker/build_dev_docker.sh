#!/bin/bash
##
## Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
##

input_arg=$1
USER_ID=$(id -u "$USER")


if [ -z "$input_arg" ]; then
    arch=$(uname -m)
    echo "Argument empty, trying to build based on architecture"
    if [ "$arch" == "x86_64" ]; then
        input_arg="x86"
    elif [ "$arch" == "arm64" ]; then
        input_arg="aarch64"
    elif [ "$arch" == "aarch64" ]; then
        input_arg="aarch64"
    fi
fi

user_dockerfile=user.dockerfile

if [[ $input_arg == *isaac_sim* ]] ; then
    user_dockerfile=user_isaac_sim.dockerfile
fi

echo $input_arg
echo $USER_ID

if [[ $input_arg == *isaac_sim* ]] ; then
    # Extract Isaac Sim version from dockerfile
    ISAAC_SIM_VERSION=$(grep "ARG ISAAC_SIM_VERSION=" isaac_sim.dockerfile | cut -d'=' -f2)
    
    # Build the base image first
    echo "Building base Isaac Sim image with version $ISAAC_SIM_VERSION"
    docker build -f isaac_sim.dockerfile --tag curobo_docker:isaac_sim .
    
    # Then build the user image
    echo "Building user Isaac Sim image with version $ISAAC_SIM_VERSION"
    docker build --build-arg USERNAME=$USER --build-arg USER_ID=${USER_ID} --build-arg IMAGE_TAG=isaac_sim -f $user_dockerfile --tag curobo_docker:user_${input_arg}_${ISAAC_SIM_VERSION} .
else
    docker build --build-arg USERNAME=$USER --build-arg USER_ID=${USER_ID} --build-arg IMAGE_TAG=$input_arg -f $user_dockerfile --tag curobo_docker:user_$input_arg .
fi