#!/bin/bash
# Script to prepare the build environment for Causal Conv1d.
#
# Example usage:
#   ./prepare_for_build.sh v1.5.4

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <causal_conv1d_version>"
    echo "Example: $0 v1.5.4"
    exit 1
fi

CAUSAL_CONV1D_VERSION=$1

# Ensure that the Causal Conv1d version is supported.
if [ ! -d "${ROOT}/build_scripts/patches/${CAUSAL_CONV1D_VERSION}" ]; then
    echo "Error: patches/${CAUSAL_CONV1D_VERSION} directory does not exist"
    exit 1
fi

# Apply patches.
for patch in "${ROOT}/build_scripts/patches/${CAUSAL_CONV1D_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done
