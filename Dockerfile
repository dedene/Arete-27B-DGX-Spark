# TensorFold's native CUDA engine, from its release archive (sha256 checked), on Mia's published image. That image already holds
# the NVIDIA PyTorch base (cuBLASLt for vision) and the pinned JPEG and WebP decoders, so nothing here is built from source.
FROM ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold@sha256:5c6c42b9747318c1cd8b6cb85cb69493fff2e82cb23b61e49b9cf54cd09ac28e
ARG TF_VERSION=1.2.3
ARG TF_SHA256=02bb060e2f42bc4dcc0d3c6800022046b10420e0237ebde2519343703f4f68f9
RUN curl -fsSL -o /tmp/tf.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_SHA256}  /tmp/tf.tar.gz" | sha256sum -c - && \
    mkdir -p /opt/tensorfold-release && tar -xzf /tmp/tf.tar.gz -C /opt/tensorfold-release --strip-components=1 && rm /tmp/tf.tar.gz && \
    test -x /opt/tensorfold-release/bin/tensorfold-native && test -s /opt/tensorfold-release/lib/libtfgrammar.so && \
    ln -sf /opt/tensorfold-release/bin/tensorfold-native /usr/local/bin/tensorfold-native
WORKDIR /workspace
