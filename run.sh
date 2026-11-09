#!/bin/bash
# --contain without binding /dev was causing some strange CPU usage whenever MuJoCo
# was running, two solutions found: not using --contain at all, or using --contain and
# binding host's /dev to container.
# /dev is also needed so the RealSense camera (/dev/video*, /dev/bus/usb, /dev/hidraw*)
# is visible inside the container.

rm -rf workdir/*

singularity run \
    --contain \
    $(if lspci | grep -qi nvidia; then echo ' --nv'; else echo ''; fi) \
    $(if [ -f "${HOME}/.gitconfig" ]; then echo ' --bind='${HOME}'/.gitconfig'; else echo ''; fi) \
    $(if [ -f "${HOME}/.vimrc" ]; then echo ' --bind='${HOME}'/.vimrc'; else echo ''; fi) \
    --bind=/etc/hosts \
    --bind=/etc/localtime \
    --workdir=workdir \
    --bind=/proc \
    --bind=/sys \
    --bind=/dev \
    --bind=scripts:/scripts \
    --bind=/run/user/${UID} \
    --home=home:${HOME} \
    --bind=$HOME/.ssh \
    SAM3D_apptainer
