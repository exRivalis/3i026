# Docker image for the POBJ (Object-Oriented Programming in Java) at UPMC, Paris, France
# Used for GitLab continuous integration with JUnit
#
# Build with: `docker build -t pobj .`
# Run with: `docker run pobj ./run.sh`
#
# - Debian based
# - install Oracle Java 8
# - install JUnit 4
# - create a 'pobj' user


FROM debian:stable

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends software-properties-common gnupg dirmngr && \
  add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
  apt-get update && \
  echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections && \
  apt-get install -y oracle-java8-installer junit4 oracle-java8-set-default && \
  adduser --disabled-password --gecos 'POBJ' pobj

COPY . /home/pobj
RUN chown -R pobj:pobj /home/pobj

USER pobj
WORKDIR /home/pobj
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN \
  javac -version && \
  java -version
  
  
