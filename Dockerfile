FROM centos:7
USER root
RUN yum -y update
RUN yum -y install readline-devel zlib-devel bzip2-devel sqlite-devel openssl-devel \
    libXext.x86_64 libSM.x86_64 libXrender.x86_64 gcc gcc-c++ libffi-devel python-devel git
RUN yum install -y https://repo.ius.io/ius-release-el7.rpm
RUN yum install -y python36u python36u-devel python36u-libs python36u-pip
RUN pip3 install pandas sklearn matplotlib statsmodels
RUN echo alias python="python3" >> ~/.bashrc
RUN echo alias pip="pip3" >> ~/.bashrc
RUN source ~/.bashrc
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
