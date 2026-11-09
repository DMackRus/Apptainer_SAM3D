bootstrap:docker
From:ubuntu:22.04

%labels

AUTHOR David Russell

%environment
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export XDG_RUNTIME_DIR=/tmp  # This fixes an issue with /run read-only permissions when using Rviz and other tools.

    # /usr/lib/x86_64-linux-gnu is listed before the --nv injected /.singularity.d/libs so the
    # container's own GL/EGL dispatch libs (glvnd) are used instead of the host's copies, which
    # can need a newer glibc than Ubuntu 22.04 has (GLIBC_2.38 errors from libEGL.so.1).
    # The NVIDIA vendor libs (libEGL_nvidia, libGLX_nvidia, libcuda) still come from the host.
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export PATH=/opt/conda/condabin:$PATH

    # Conda lives in /opt/conda (read-only at runtime), so environments and the
    # package cache are kept in the writable home folder.
    export CONDA_ENVS_DIRS=$HOME/conda/envs
    export CONDA_PKGS_DIRS=$HOME/conda/pkgs

    export PYTHONUNBUFFERED=1

%files
  scripts /scripts

%post
  /scripts/post_script.sh

%runscript
  exec /bin/bash "$@" --rcfile /scripts/run_script.sh
