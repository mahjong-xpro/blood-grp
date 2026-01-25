# syntax=docker/dockerfile:1.4-labs

FROM archlinux:base-devel as libblood_build

RUN <<EOF
pacman -Syu --noconfirm --needed rust python
pacman -Scc
EOF

WORKDIR /
COPY Cargo.toml Cargo.lock .
COPY libblood libblood

RUN cargo build -p libblood --lib --release

# -----
FROM archlinux:base

RUN <<EOF
pacman -Syu --noconfirm --needed python python-pytorch python-toml python-tqdm tensorboard
pacman -Scc
EOF

WORKDIR /mortal
COPY mortal .
COPY --from=libblood_build /target/release/libblood.so .

ENV MORTAL_CFG config.toml
COPY <<'EOF' config.toml
[control]
state_file = '/mnt/mortal.pth'

[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
EOF

VOLUME /mnt

ENTRYPOINT ["python", "mortal.py"]
