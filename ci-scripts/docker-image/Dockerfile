FROM ubuntu:16.04
RUN apt-get -y update && apt-get install -y curl git python python-pip cmake pkg-config libx11-dev libgl1-mesa-dev libfontconfig1-dev software-properties-common
RUN add-apt-repository -y -u ppa:glasen/freetype2
RUN apt-get -y install freetype2-demos
RUN pip install mako voluptuous PyYAML servo-tidy
RUN useradd -d /home/worker -s /bin/bash -m worker
USER worker
WORKDIR /home/worker
ENV PATH $PATH:/home/worker/.cargo/bin
CMD /bin/bash
