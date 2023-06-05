FROM pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9

ENTRYPOINT []

# install GLIP & pretrained models
# can not get pretrained models from https://huggingface.co
# so we packed all models to a tar file with following structure:
#    bert-base-uncased/
#    bert-base-uncased/tokenizer.json
#    bert-base-uncased/vocab.txt
#    bert-base-uncased/config.json
#    bert-base-uncased/tokenizer_config.json
#    bert-base-uncased/pytorch_model.bin
#    MODEL/
#    MODEL/glip_a_tiny_o365.pth
#    MODEL/swin_tiny_patch4_window7_224.pth
# when rebuild this docker image, you can get models from /app/MODELS and /app/bert-base-uncased in container
ADD ./pretrained/ymir-glip-models.tar.gz /app
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
# Change the pip source if needed
# RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
# RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers loralib==0.1.1
COPY ./configs /app/configs
COPY ./knowledge /app/knowledge
COPY ./maskrcnn_benchmark /app/maskrcnn_benchmark
COPY ./odinw /app/odinw
COPY ./setup.py /app/setup.py
COPY ./tools /app/tools
RUN cd /app && python setup.py build develop --user && cd /

# setup ymir & ymir-GLIP
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.4.0"

COPY ./ymir /app/ymir
COPY ./start.py /app/start.py
RUN mkdir /img-man && mv /app/ymir/img-man/*.yaml /img-man/

ENV PYTHONPATH=.
WORKDIR /app
RUN echo "python3 /app/start.py" > /usr/bin/start.sh
CMD bash /usr/bin/start.sh
