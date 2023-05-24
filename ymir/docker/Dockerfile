FROM pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9

ENTRYPOINT []

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.4.0"

COPY . /app
RUN mkdir /img-man && mv /app/ymir/img-man/*.yaml /img-man/

ENV PYTHONPATH=.

WORKDIR /app
RUN echo "python3 /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh