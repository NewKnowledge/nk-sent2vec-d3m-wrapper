# FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-devel-20180419-092215
FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.6.5

ENV HOME=/app

WORKDIR $HOME

# install nk_sent2vec
COPY . $HOME/
# RUN python3 setup.py install 
RUN pip3 install .

# check that it runs by triggering tests
CMD python3 nk_sent2vec_wrapper/wrapper.py
# CMD nosetests 

